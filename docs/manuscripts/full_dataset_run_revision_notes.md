# Full-dataset run: workflow notes

**Run**: `publication_full_dataset_distributed_20260526_short_hp1`\
**Artifact root**:
`artifacts/publication_full_dataset_distributed_results/publication_full_dataset_distributed_20260526_short_hp1/artifacts/final_manuscript_artifacts/`

______________________________________________________________________

## Implemented workflow

**Dataset**: 30,000 runs (28,500 training, 1,500 holdout, 5% holdout fraction, seed 123).
158 scalar LHC inputs × 4 binary scenario combinations.

**Pipeline stages and verified counts:**

| Stage                        | Quantity                      | Value                |
| ---------------------------- | ----------------------------- | -------------------- |
| Output conditioning          | Retained outputs for PCA      | 9,954 of 23,495      |
| Output conditioning          | PCA components (90% variance) | 20                   |
| Empirical null screening     | Retained scalar inputs        | 69 of 158            |
| Interaction discovery        | Retained pairs                | 62                   |
| Nonlinear discovery          | Retained transforms           | 41                   |
| Enriched design matrix       | Total columns                 | 172 (69 + 62 + 41)   |
| LASSO + stability            | Stable features               | 172 of 172           |
| HC3 inferential filter       | Retained features             | 172 of 172           |
| Delta-threshold pruning      | Pruned features               | 40 (threshold 0.002) |
| Final OLS support            | Total predictors              | 132                  |
| Final OLS support            | Main-effect terms             | 54                   |
| Final OLS support            | Interaction terms             | 49                   |
| Final OLS support            | Nonlinear transforms          | 29                   |
| Final OLS coefficient matrix | Outputs covered               | 23,495 (all)         |
| Final OLS nRMSE              | Holdout macro (Y-train range) | 0.0721               |

**nRMSE scope**: metric evaluated on 9,954 outputs with non-negligible variance.
Near-constant culled outputs are in the coefficient matrix with near-zero coefficients.

______________________________________________________________________

## Ablation results

| Model stage      | Features | nRMSE  | CI lower | CI upper |
| ---------------- | -------- | ------ | -------- | -------- |
| Null mean        | 0        | 0.1653 | 0.1635   | 0.1663   |
| Main effects OLS | 158      | 0.0812 | 0.0796   | 0.0821   |
| Screened OLS     | 69       | 0.0812 | 0.0796   | 0.0821   |
| Penalized OLS    | 172      | 0.0709 | 0.0694   | 0.0719   |
| Final OLS        | 132      | 0.0721 | 0.0706   | 0.0730   |

Source: `tables/ablation_table.csv`.

______________________________________________________________________

## Per-output nRMSE distribution (9,954 outputs)

| Statistic | nRMSE |
| --------- | ----- |
| p10       | 0.018 |
| p25       | 0.036 |
| p50       | 0.061 |
| p75       | 0.093 |
| p90       | 0.141 |

Worst three outputs: `OI.OHC output by product[TransEster, P, diesel]`
for 2023 (0.495), 2024 (0.487), 2025 (0.476).

Source: `tables/per_output_nrmse_summary.csv`.

______________________________________________________________________

## Feature composition (final 132 predictors by module)

| Module                      | Features | Share |
| --------------------------- | -------- | ----- |
| Cellulosic Hydrocarbons     | 44       | 33%   |
| Wet Waste Hydrocarbons      | 36       | 27%   |
| Oil Hydrocarbons            | 28       | 21%   |
| Starch Ethanol Hydrocarbons | 13       | 10%   |
| Algal Hydrocarbons          | 11       | 8%    |

Source: `figures/figure_selected_by_module_data.csv`.

______________________________________________________________________

## Open items

- **[PENDING]** Job 14043519 (`final_manuscript_artifacts` rerun): regenerates
  `coefficient_matrix_standardized.csv` and `coefficient_matrix_raw_scale.csv` with all
  23,495 output rows. After completion, re-sync with
  `scripts/kestrel/collect_publication_full_dataset_distributed.sh`.
- **[PENDING]** Sensitivity study §7 results (jobs running on Kestrel).
- **[PENDING]** Repository DOI, supplement archive DOI, acknowledgment/disclaimer text.
- **[PENDING]** Investigate worst outputs (`OI.OHC output by product[TransEster, P, diesel]`
  2023–2025, nRMSE ~0.476–0.495) before submission.

______________________________________________________________________

## Artifact collection command

```bash
bash scripts/kestrel/collect_publication_full_dataset_distributed.sh \
  --study-id publication_full_dataset_distributed_20260526_short_hp1 \
  --study-root /scratch/dhetting/bsm/studies/publication_full_dataset_distributed_20260526_short_hp1
```
