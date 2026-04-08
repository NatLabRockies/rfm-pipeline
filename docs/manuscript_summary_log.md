# Running manuscript-summary log (initial audit)

Only values directly observed in the uploaded archive or stored notebook outputs are listed here.

- Upstream screening dataset size observed in `null_distribution.py`: 300000 samples.
- Upstream screening method: SALib delta sensitivity with permutation null.
- Upstream null settings: `B=200`, `alpha=0.05`, method `fwer-max`.
- Upstream influential-resampling size: 2000 samples.
- Downstream modeling dataset loaded in notebook: 20000 rows.
- Downstream candidate input count loaded in notebook: 352.
- Downstream full output count loaded in notebook: 23495.
- Diagnostic plotting subset in notebook: 555 outputs (15 variables x 37 years).
- Notebook external holdout split: 18000 train / 2000 holdout.
- Notebook culling prior to PCA-LASSO search: 9782 outputs kept, 13713 outputs culled.
- Hyperparameter grid in notebook search: `l1_ratio in {1.0, 0.95, 0.9}`, `alpha_frac in {0.75, 0.5, 0.25, 0.10, 0.05, 0.02, 0.01}`.
- Notebook-selected screening hyperparameters from stored output: `l1_ratio=1.00`, `alpha_frac=0.10`, `alpha_abs≈2.8541`, criterion `EBIC`.
- Notebook selected feature count after final LASSO stage: 346.
- Notebook selected-feature structure counts: 62 first-order, 40 nonlinear transformations, 244 second-order interactions.
- Notebook transformation counts shown in stored output: 38 quadratic, 2 inverse.
- Archived multivariate script default holdout: 5% (diverges from notebook 10%).
- Archived multivariate script tuning subset file: `tune_vars.csv`.
- Null-model nRMSE baseline in the notebook remains notebook-derived until confirmed by a source script.
- Bootstrap CI design present in notebook: row bootstrap, percentile interval, fixed `Y_ref` range normalization.
- Stored notebook code default for bootstrap replicates: 1000.

- Engineering contract for refactor: Pixi-managed environment, Ruff for Python/notebook code, mdformat for Markdown, Sphinx package docs, GitHub Actions CI, NumPy-style docstrings on public APIs.


- **Resolved subset provenance:** the 20,000 modeling rows are a balanced stratified sample from the 300,000 raw runs, with 5,000 sampled separately within each boolean scenario combination defined by AFSC and UAEORO.
- **Resolved screening provenance:** the canonical null/screening workflow is the recovered `null_distribution.py` script (SALib Delta sensitivity with permutation null cutoffs), accessed in the refactor through a stable adapter.
- **Important distinction:** any later mean-only or intercept-only regression baseline used for nRMSE reporting should be labeled as a regression baseline, not as the canonical upstream null-screening workflow.
