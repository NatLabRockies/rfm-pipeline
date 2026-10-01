# Interpreting results

Interpret a run from its recorded contract and baselines, not from counts copied
from another dataset.

## Confirm completion first

For the canonical API bundle:

1. Confirm `manifest.json` exists.
1. Load tables with `load_postfit_bundle(...)`.
1. Confirm the manifest's feature/output counts and file map match the files.

For the staged workflow:

1. Require `run_complete.json` for the requested stage window.
1. Read each completed stage's `*_summary.csv`.
1. If `reproduction_audit/` exists, require `qa_status == "pass"` and no
   failed metric checks.
1. Treat `run_failed.json`, `run_interrupted.json`, or
   `run_abandoned.json` as incomplete runs.

## Read the holdout metric

Macro nRMSE is the mean, across eligible outputs, of:

```text
holdout RMSE / training-reference output range
```

Outputs with a training range below the configured minimum are excluded. The
bootstrap interval resamples holdout rows while keeping training ranges fixed.

Lower nRMSE is better, but there is no universal pass threshold. Compare it
with:

- a prespecified baseline evaluated on the same holdout;
- the uncertainty interval;
- per-output metrics and exclusion reasons; and
- the intended scientific use.

Do not compare nRMSE values that use different output sets, normalization
references, or holdout definitions.

## Inspect model size and stability

- `selected_input_metadata` shows what the canonical screen retained.
- `retained_terms.csv`, `retained_interaction_pairs.csv`, and
  `retained_transformations.csv` show staged discovery decisions.
- `final_stable_support.csv` records stability-selected terms.
- `coef_matrix_raw_scale` or the final model coefficient matrix records the
  fitted linear surrogate.

Zero retained features is a hard failure for the canonical workflow. A large
or small support is not inherently wrong; judge it against sample size,
candidate-library size, resampling stability, and holdout performance.

## Check provenance before comparing runs

Record at least:

- package commit or version;
- input data and split identity;
- configuration file and random seed;
- output eligibility/exclusion policy;
- artifact manifest or hashes; and
- whether execution was local, out-of-core, or distributed.

Counts and thresholds from another study are not acceptance ranges for a new
dataset.

## Common warning signs

| Observation                                     | Check                                                                  |
| ----------------------------------------------- | ---------------------------------------------------------------------- |
| Non-finite nRMSE                                | Empty holdout, non-finite inputs/outputs, or no eligible output ranges |
| No retained feature                             | Screening strength, signal, alignment, and candidate features          |
| Very wide interval                              | Holdout size, outliers, and stratification                             |
| Different results with the same inputs          | Seed, package revision, config, data order, and parallel backend       |
| Stage directory exists but no completion marker | Interrupted or failed execution                                        |

Continue with [Troubleshooting](troubleshooting.md) for corrective steps or
[Artifact reference](artifact_reference.md) to locate a file.
