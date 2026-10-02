# Overview

`rfm-pipeline` provides one primary fitting path:
`run_canonical_workflow(...)`. Give it aligned training and holdout DataFrames;
it returns the screening result, fitted OLS model, evaluation summary, and
portable artifact tables.

## Canonical API flow

```text
aligned train/holdout DataFrames
        |
        v
multitask elastic-net screening
        |
        v
output-wise OLS fit
        |
        v
holdout macro nRMSE + bootstrap interval
        |
        v
manifest + portable coefficient tables
```

The workflow screens the columns it receives. It does not decide which domain
transformations or interactions are scientifically appropriate. Create those
columns before fitting, or use `apply_feature_expansion(...)` with an explicit
`FeatureExpansionSpec`.
