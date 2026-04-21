# BSM reduced-form workflow audit

This page records the recovered workflow stages, the strongest available provenance for each
stage, and the main divergences between archived scripts and recovered notebooks.

## Recovered source and notebook artifacts

The audited archive contains four distinct workflow fragments:

1. `null_distribution.py`

   - canonical upstream screening script
   - operates on `X_standardized.nl.300000.npy` and `Y_standardized.300000.npy`
   - uses SALib Delta sensitivity with a permutation null (`B=200`, `alpha=0.05`,
     `method="fwer-max"`)
   - produces influential-input reports and new LHS samples of size `N_NEW = 2000`

1. `make_nonlinear_features.ipynb`

   - canonical recovered feature-expansion artifact for interactions and nonlinear transforms
   - adds scenario flags `AFSC` and `UAEORO`
   - reads `influential_factors.300k.null_010.nl.csv` and creates `sa_068.null_010.X.nl.csv`

1. `multivariate_mmreg_pipeline.with_subset.py`

   - archived regularized multi-output regression script
   - uses `MultiTaskElasticNetCV` on a holdout split
   - tunes on a target subset from `tune_vars.csv`, then refits on all outputs
   - uses a 5% holdout by default unless overridden

1. `LASSO_to_OLS_v9.ipynb`

   - recovered notebook workflow for PCA/de-biased-LASSO/final-OLS modeling
   - uses a 10% external holdout and adds bootstrap confidence intervals for macro nRMSE
   - exports a richer artifact schema than the archived script

## Main script-versus-notebook divergences

### Upstream screening stage

- The notebook does not implement the Delta permutation-null screen from
  `null_distribution.py`.
- It begins after an already reduced-and-expanded `X` exists.
- The canonical upstream null-screening implementation is therefore still the recovered
  source script, accessed in the package through `bsm_rfm.null_screening`.

### Modeling subset creation

- `null_distribution.py` works on `300000` standardized samples.
- The recovered downstream modeling path uses a balanced 20,000-row subset.
- The subset provenance is now resolved: 5,000 rows were sampled independently inside each
  AFSC/UAEORO boolean combination and then recombined.

### Holdout design

- `multivariate_mmreg_pipeline.with_subset.py` defaults to `test_size=0.05`.
- `LASSO_to_OLS_v9.ipynb` uses a deterministic 10% external holdout (`18000` train,
  `2000` holdout) with scenario stratification when available.
- That is a real scientific divergence and should remain explicit in package documentation.

### Regularized screening model

- Archived script: `MultiTaskElasticNetCV`, direct multi-output fit on scaled `Y`.
- Notebook: PCA compression of `Y`, custom alpha-path search, de-biased LASSO, and
  final OLS.
- These are materially different workflows. The package currently implements a tested
  multitask-elastic-net screening foundation rather than a full PCA/de-biased-LASSO port.

### Evaluation and export

- Archived script reports point-estimate RMSE and nRMSE.
- The recovered notebook adds a null mean-prediction regression baseline and bootstrap
  confidence intervals for macro nRMSE.
- The package now implements bootstrap macro nRMSE reporting and canonical post-fit bundle
  assembly at the workflow layer.

## Current package interpretation

The live package organizes the workflow into explicit stages:

1. upstream Delta null screening
1. feature expansion
1. balanced modeling-subset creation
1. regularized screening
1. final OLS fitting
1. evaluation and export
1. downstream visualization

The screening/final-fit/evaluation/export path is implemented directly in the package.
The upstream Delta screen remains source-derived through an adapter, and the recovered
feature-expansion specification remains only partially promoted into a canonical default.
