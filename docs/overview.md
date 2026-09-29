# Overview

`rfm-pipeline` is a reusable modeling package. The BSM analysis is one
case study, not the package's default identity.

## Pick an execution surface

| Surface            | Use it when                                                                                                                  | Entry point                                          |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Canonical API      | You have aligned train/holdout DataFrames and want a compact, exportable linear surrogate                                    | `run_canonical_workflow(...)`                        |
| Staged workflow    | You need output PCA, empirical-null screening, interaction/nonlinear discovery, stability selection, and resumable artifacts | `tools/run_manuscript_pipeline.py`                   |
| Distributed stages | A stage must be sharded and reduced on a scheduler                                                                           | `rfm-hpc-submit`, `rfm-hpc-worker`, `rfm-hpc-reduce` |
| BSM case study     | You want a concrete, fully tracked application                                                                               | `examples/bsm-manuscript/`                           |

Some advanced modules and filenames retain `manuscript` in their names for
compatibility. That label identifies their historical origin; it does not make
the core API publication-specific.

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

The canonical API does not automatically perform upstream permutation-null
screening or feature expansion. Supply already prepared feature columns, or
use the staged workflow when those steps are required. See
[Workflow scope boundary](scope_boundary.md) for the exact implementation
boundary.

## Repository boundaries

- `rfm-pipeline`: generic workflow code, docs, tests, and case studies.
- `bsm-public-rf`: ready-to-use BSM coefficients and prediction API.
- `bsm-public-rf-manuscript`: article and submission source.

Keeping these responsibilities separate lets model users avoid the research
workflow and lets workflow users start from a neutral example.
