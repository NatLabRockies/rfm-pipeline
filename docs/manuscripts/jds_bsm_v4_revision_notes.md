# JDS BSM manuscript revision notes

## What was changed

- Addressed all manuscript TODOs that could be resolved from the uploaded scripts/notebooks and bibliography.
- Rewrote the workflow to reflect the recovered implementation rather than the earlier generic regression-screening description.
- Replaced the old marginal-regression empirical-null description with the script-verified permutation-calibrated Delta sensitivity screen.
- Added implementation details recovered from the notebooks: 20,000 total runs; 18,000 training rows; 2,000 scenario-stratified holdout rows; 23,495 scalar outputs; 352 materialized enriched predictors; 9,782 filtered outputs for PCA support recovery; 13,713 culled from the PCA support-recovery stage; 95% PCA variance target; 346 LASSO-selected features; 340 final OLS-retained predictors.
- Defined the nRMSE calculation from the scripts.
- Added placeholder ablation and output-wise performance tables for the analyses you said to table until numbers are produced.
- Revised interpretability claims so they refer to a finite, named, algebraic, exportable support and coefficient artifact, not a manually small coefficient table.
- Removed the final post-selection interval filter language and replaced it with a more defensible de-biased-LASSO-as-selection-device description.
- Added verified citation coverage for Tree SHAP / SHAP interaction values, BSM documentation, Sobol-style interaction structure, stability selection, and de-biased LASSO inference.
- Cleaned the bibliography to cited entries only and removed duplicate-equivalent unused entries.
- Added a workflow-paper positioning paragraph and a sensitivity-analysis complementarity paragraph.
- Rewrote reproducibility as though the repository and DOI are available, using placeholder URLs/DOIs.
- Rewrote the conclusion in a more direct technical voice.
- Removed advocacy-sensitive “policy” language from the revised manuscript text.

## Recovered true workflow

1. Generate four scenario/boolean combinations with 5,000 sampled runs each, producing 20,000 total runs.
1. Read the 20,000-run enriched input matrix and 23,495-output response matrix.
1. Convert X and Y to numeric float64.
1. Create a deterministic scenario-stratified 10% external holdout split before adaptive modeling.
1. Use 18,000 training rows for all filtering, standardization, feature discovery, LASSO tuning, de-biased support recovery, and final OLS fitting.
1. Use 2,000 holdout rows only for final external validation metrics.
1. Use train-only output culling for PCA support recovery: 9,782 outputs retained, 13,713 culled.
1. Use PCA on standardized filtered outputs with a 95% variance-retention target.
1. Tune a multi-task elastic-net/LASSO path by EBIC with gamma = 0.5.
1. Run final LASSO support recovery using de-biased coefficient screening and BH-FDR at alpha = 0.05.
1. Fit final OLS on the full original output matrix using selected features, an intercept, and train-only standardization metadata.
1. Use automatic HC3 covariance for OLS diagnostics after a Breusch--Pagan screen.
1. Export raw-scale and standardized-scale coefficient products, intercepts, metadata, and validation summaries.

## nRMSE found in scripts

The current final OLS notebook defines `macro_nrmse_with_ref(Y_true, Y_pred, Y_ref, min_range=1e-6)` as:

- compute RMSE separately for each output over evaluation rows;
- compute the normalization range for each output from `Y_ref`;
- keep only outputs whose reference range is at least `1e-6`;
- divide each output RMSE by its reference range;
- average those normalized RMSE values across retained outputs.

For the final OLS holdout metric, `Y_ref` is the training response matrix. Thus the manuscript now defines holdout macro nRMSE as a mean of output-wise holdout RMSE values normalized by training-set output ranges.

## Remaining gaps / placeholders

- The figure image files were not included, so I updated captions based on the visualization scripts but could not inspect the rendered figures.
- Ablation metrics still need to be produced and inserted.
- Output-wise nRMSE quantiles and worst-output diagnostics still need to be produced and inserted.
- The exact larger “26,560 candidate library” convention was not recoverable from the scripts as a materialized design matrix. The scripts verify the materialized 352-predictor enriched matrix: 63 first-order + 248 second-order + 41 one-variable transformations. I removed the 26,560 result from the manuscript body rather than treating it as verified.
- The repository URL, DOI, supplement filename, names of internal reviewers/collaborators, award number, contract text, and required lab disclaimer remain placeholders.
- The Steve Peterson affiliation remains a placeholder because it was not recoverable from the uploaded files.

## Advocacy-sensitive language flagged

The revised manuscript removes or avoids the original policy-oriented phrasing. The remaining language is framed around scenario analysis, scientific analysis, simulator reduction, validation, and reusable statistical artifacts rather than policy claims or recommendations.
