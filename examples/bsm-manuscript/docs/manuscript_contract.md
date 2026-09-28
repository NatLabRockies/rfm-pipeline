# BSM Manuscript Case Study Contract

This document describes the case study data, modeling scope, and validation criteria
for the BSM reduced-form model manuscript.

See `configs/manuscript_case_study.yml` for the machine-readable contract specification.

## Key Statistical Design Decisions

### Permutation Testing (B+1 Design)

The manuscript reports **B = 200** permutations for the empirical null screening stage.
The implementation uses **B + 1 = 201** permutations. This is the standard Monte Carlo
exact p-value design:

- 1 observed test statistic
- B = 200 null draws (label permutations)

The exact Monte Carlo p-value is computed as:

  p = (number of null statistics ≥ observed statistic + 1) / (B + 1)

Using B+1 in the denominator ensures a valid, exact p-value bound even for the most
extreme case (observed statistic exceeds all null draws). The manuscript correctly
reports B=200 null draws; the "+1" is a computational convention, not an additional
null draw. See the config comment in `configs/hpc/kestrel_publication_full_dataset.yml`
and `configs/manuscript_case_study.yml`.

### Significance Thresholds

- Empirical null FDR: Benjamini-Hochberg `q = 0.05`
- Interaction discovery p-threshold: `0.05`
- Nonlinear discovery: EDF threshold ≥ 2.5

### Resampling Design

- Sparse selection uses **50 subsamples** of 80% of rows (without replacement, seed=123)
- Jaccard stability threshold: 0.75
- Spearman rank-correlation threshold: 0.90

### Outcome of the Publication Run

- **69 terms** retained after empirical null FDR screening (bh_q=0.05)
- **41 terms** identified as nonlinear (EDF ≥ 2.5)
- Interaction discovery used 250 random forest trees, max depth 5
- Final artifacts: 100 bootstrap iterations, α=0.05
