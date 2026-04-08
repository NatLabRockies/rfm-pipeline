# BSM reduced-form workflow audit (initial)

## What is actually present in the uploaded archive
The archive contains four distinct workflow fragments:

1. `null_distribution.py`
   - Canonical upstream screening script.
   - Operates on `X_standardized.nl.300000.npy` and `Y_standardized.300000.npy`.
   - Uses SALib delta sensitivity with a permutation null (`B=200`, `alpha=0.05`, `method="fwer-max"`).
   - Produces influential-input reports and new LHS samples of size `N_NEW = 2000`.

2. `make_nonlinear_features.ipynb`
   - Canonical feature-expansion artifact for interactions/nonlinear transforms.
   - Adds scenario flags `AFSC` and `UAEORO`.
   - Reads `influential_factors.300k.null_010.nl.csv` and creates `sa_068.null_010.X.nl.csv`.

3. `multivariate_mmreg_pipeline.with_subset.py`
   - Canonical regularized multi-output regression script in the archive.
   - Uses `MultiTaskElasticNetCV` on a holdout split.
   - Tunes on a target subset from `tune_vars.csv`, then refits on all outputs.
   - Uses a 5% holdout by default, though the provided shell call does not override that default.

4. `LASSO_to_OLS_v9.ipynb`
   - Notebook-only fused workflow that is not backed by an equivalent standalone script in the archive.
   - Replaces the archive script's `MultiTaskElasticNetCV` workflow with a PCA-on-Y + de-biased LASSO + final OLS pipeline.
   - Uses a 10% external holdout and adds bootstrap CIs for macro nRMSE.
   - Exports a much richer artifact schema than the archived script.

## Direct notebook-versus-script divergences

### Upstream screening stage
- The notebook does **not** implement the delta/permutation-null screening from `null_distribution.py`.
- Instead, it assumes an already reduced-and-expanded `X` (`sa_068.null_010.X.nl.parquet`) exists.
- Therefore the notebook begins **after** the actual null-screening script.

### Training sample derivation
- `null_distribution.py` works on `300000` standardized samples.
- `collect_N_samples.py` randomly samples **files** from a directory and concatenates them; it does not document a stratified row subsample.
- The notebook loads `20000` rows directly and reports `raw X shape: (20000, 352)` and `raw Y shape: (20000, 23495)`.
- This means the 20k modeling set is downstream of the 300k screening stage, but the exact canonical subset-generation script is **not fully recovered** in this archive.

### Holdout design
- `multivariate_mmreg_pipeline.with_subset.py` defaults to `test_size=0.05`.
- `LASSO_to_OLS_v9.ipynb` uses a deterministic 10% external holdout (`18000` train, `2000` holdout) and stratifies by scenario when available.
- This is a scientifically meaningful divergence and must become an explicit configuration choice in the refactor.

### Regularized screening model
- Archived script: `MultiTaskElasticNetCV`, direct multi-output fit on scaled `Y`.
- Notebook: PCA compression of `Y`, custom alpha-fraction path search, EBIC/ALO scoring, KKT check, de-biased LASSO, BH-FDR rowwise selection.
- These are materially different scientific workflows; the notebook is not a thin wrapper over the script.

### Null model / baseline reporting
- Archived script reports RMSE/nRMSE point estimates only.
- Notebook adds a null mean-prediction baseline and row-bootstrap percentile CIs for macro nRMSE.
- There is no evidence in the archive that the notebook's null baseline was part of the original production script, so this baseline remains notebook-derived until a source script confirms it.

### Export schema
- Archived script exports coefficients, intercepts, holdout predictions, and summary metrics.
- Notebook exports richer metadata: all-input order, selected-input order, retained-input order, output order, transform vectors, raw- and standardized-scale coefficients, and postfit diagnostics.
- The visualization notebook already expects the richer notebook-style export contract.

## Additional gaps / defects found
- `collect_N_samples.sh` calls `collect_N_samples.py` with `--out`, but the Python script expects `--output`.
- The archive does not include a canonical standalone script for the PCA/debiased-LASSO/final-OLS workflow now embodied in `LASSO_to_OLS_v9.ipynb`.
- The archive does not include the exact script that generated the final 20k modeling subset from the upstream 300k stage in a provenance-rich way.

## Recovered case-study numbers so far
- Upstream null-screening sample size: `300000`.
- Downstream modeling matrix loaded by the notebook: `20000` rows.
- Notebook train/holdout split: `18000 / 2000`.
- Notebook candidate input count at model start: `352`.
- Notebook full output count: `23495`.
- Diagnostic plotting subset: `555` outputs.
- Notebook output culling before PCA-LASSO search: `9782` kept, `13713` culled.
- Notebook selected feature count after final LASSO stage: `346`.
- Parsed selected-feature counts from notebook output: `62` first-order, `40` nonlinear, `244` second-order.
- Upstream null permutation count in `null_distribution.py`: `B=200`.
- Upstream influential-resampling size in `null_distribution.py`: `N_NEW=2000`.

## Refactor implication
The refactor should be organized as a staged workflow with explicit provenance boundaries:
1. upstream delta/null screening,
2. feature expansion,
3. modeling subset creation,
4. regularized screening,
5. final OLS,
6. evaluation/export,
7. downstream visualization.

The notebook-derived stages can be refactored, but they must be labeled as notebook-derived unless or until an equivalent source script is recovered.


## Audit update: recovered subset path and null-screening provenance

The 20,000-row modeling set provenance is now resolved. The canonical subset-generation path from the 300,000-run archive was **stratified random sampling by the two boolean inputs**: draw 5,000 run numbers independently within each of the four AFSC/UAEORO combinations, then recombine the four strata into a balanced 20,000-row modeling set. This overrides the earlier placeholder assumption that only a generic random subset script was available.

The canonical null/screening stage is the recovered `null_distribution.py` script, not the notebook-derived mean-baseline logic. In the refactor scaffold, null-screening calculations are therefore delegated through a stable adapter to the recovered source workflow so the permutation-null Delta implementation remains source-of-truth.

A separate mean-prediction baseline may still be reported later as a regression baseline for nRMSE comparison, but it is **not** the canonical upstream null-screening implementation.
