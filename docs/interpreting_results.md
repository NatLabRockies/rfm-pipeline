# Interpreting results

Interpret a run from its recorded contract and baselines, not from counts copied
from another dataset.

## Confirm completion first

1. Confirm `manifest.json` exists.
1. Load tables with `load_postfit_bundle(...)`.
1. Confirm the manifest's feature/output counts and file map match the files.

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
- `coef_matrix_raw_scale` records the fitted linear surrogate in raw units.

Zero retained features is a hard failure for the canonical workflow. A large
or small support is not inherently wrong; judge it against sample size,
candidate-library size, coefficient behavior, and holdout performance.

## Check provenance before comparing runs

Record at least:

- package commit or version;
- input data and split identity;
- fitting options and random seeds;
- output eligibility/exclusion policy;
- artifact manifest or hashes; and
- package and dependency versions.

Counts and thresholds from another study are not acceptance ranges for a new
dataset.

## Common warning signs

| Observation                            | Check                                                                  |
| -------------------------------------- | ---------------------------------------------------------------------- |
| Non-finite nRMSE                       | Empty holdout, non-finite inputs/outputs, or no eligible output ranges |
| No retained feature                    | Screening strength, signal, alignment, and candidate features          |
| Very wide interval                     | Holdout size, outliers, and stratification                             |
| Different results with the same inputs | Seed, package revision, fit options, and data order                    |

Continue with [Troubleshooting](troubleshooting.md) for corrective steps or
[Artifact reference](artifact_reference.md) to locate a file.
