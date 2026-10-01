# Overview

`rfm-pipeline` offers a compact fitting API and an advanced staged workflow.
Choose the smallest surface that includes the modeling steps you need.

## Pick an execution surface

| Surface            | Use it when                                                                                                                  | Entry point                                          |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Canonical API      | You have aligned train/holdout DataFrames and want a compact, exportable linear surrogate                                    | `run_canonical_workflow(...)`                        |
| Staged workflow    | You need output PCA, empirical-null screening, interaction/nonlinear discovery, stability selection, and resumable artifacts | `tools/run_manuscript_pipeline.py`                   |
| Distributed stages | A validated stage must be sharded and reduced on a scheduler                                                                 | `rfm-hpc-submit`, `rfm-hpc-worker`, `rfm-hpc-reduce` |

Some advanced modules, stage directories, and commands retain `manuscript` in
their names for compatibility. The canonical API does not use those surfaces.

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

The canonical API does not perform upstream permutation-null screening or
feature expansion. Supply prepared feature columns, or use the staged workflow
when those steps are required. See [Workflow scope boundary](scope_boundary.md)
for the exact implementation boundary.
