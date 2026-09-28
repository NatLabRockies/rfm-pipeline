# Per-output OLS intercepts

The released coefficient matrices (`coefficient_matrix_raw_scale.csv` and
`coefficient_matrix_standardized.csv`) deliberately omit a free intercept
column because all candidate features in the final design are mean-centred
(see `x_standardization.csv`). The per-output intercept therefore collapses
to the per-output training-set mean of the response.

To make the prediction equation fully self-contained, the same vector is
exported as `per_output_intercepts.csv` for downstream consumers that prefer
an explicit intercept term. The values are identical to the `mean` column of
`y_standardization.csv`. The `scale` column of that file is `1.0` for the
~13.2k outputs that were not response-standardized during fitting and equals
the per-output training standard deviation for the remaining outputs;
`coefficient_matrix_raw_scale.csv` absorbs that `scale_y` factor so that
predictions are recovered on the raw response scale.

The full prediction rule using the raw-scale coefficient matrix is

```
y_hat[output i, row r] = intercept[i]
                       + sum_j  coef_raw_scale[i, j] * (X_raw[r, j] - mean_x[j])
```

where `coef_raw_scale` is read from `coefficient_matrix_raw_scale.csv` and
`mean_x` is read from `x_standardization.csv`. Only mean-centring of `X` is
required; `coef_raw_scale` already absorbs both the per-feature `scale_x` and
any per-output `scale_y`.
