# Workflow ↔ Manuscript Alignment Plan

**Purpose.** Make the *generic* `rfm-pipeline` workflow genuinely implement the
methodology the JDS manuscript describes, so a future case-study rerun can
honestly reproduce the paper's claims. This plan is the root-cause response to
the JDS adversarial review (findings F1–F7, M1–M9): the executed pipeline
diverged from the principled workflow the manuscript describes, so the workflow
must be corrected — not the manuscript watered down.

**Hard constraints (apply to every slice).**

- The pipeline is a *general* reduced-form modeling workflow. It must stay 100%
  case-study-agnostic. **Never** hardcode BSM/AEO specifics or any case-study
  literal into `src/`. Capabilities are generic; tests use synthetic data.
- Use repo-local Pixi for all tooling (`pixi run ...`). No bare
  `python`/`pytest`/`ruff`.
- Focused TDD: write the failing alignment test first, observe red, then
  implement the minimal robust fix.
- Do not weaken existing tests. Do not add compatibility shims.
- Each slice edits **only** the files in its "Files to create/modify" list. If
  you need something from another slice, STOP and report a blocker.

**Validation contract.** Each slice ships a pytest module under
`tests/alignment/` with test functions named `test_<SLICE_ID_UNDERSCORED>_*`
(e.g. slice `P0-S01` → `test_P0_S01_*`). The runner validates with
`bash scripts/slice_validate.sh <SLICE-ID>`, which runs
`pixi run python -m pytest tests/alignment -k <SLICE_ID_UNDERSCORED>`.
A slice is done only when its alignment tests pass **and**
`pixi run ruff check <changed files>` is clean.

**Out of slice-runner scope (downstream, tracked separately).** Actual HPC
reruns of the case study, regeneration of manuscript numbers, public
release/DOI minting (F7 releases), AEO/BSM data-provenance capture (M8),
sensitivity-design re-execution (M1/M2/M7), and manuscript prose edits (M9,
abstract/cover rewrites) are *not* slice-runner slices — they require HPC, data
rights, or LaTeX edits outside this generic repo. This plan delivers the
generic *code capabilities* those downstream steps depend on.

______________________________________________________________________

## PHASE P0 — methodology correctness

### Slice P0-S01: Generic categorical/block predictor inputs in config + schema (F1)

**Phase:** P0
**Depends on:** none
**Estimated size:** medium
**Files to create/modify:**

- src/rfm_pipeline/config.py
- tests/alignment/test_P0_S01_categorical_inputs.py

**Context.** F1: the exported model dropped the two binary scenario switches, so
it returns identical predictions across scenarios and cannot represent scenario
effects. Root cause in generic code: the config/interface has no first-class
notion of a *categorical/block predictor* input that must flow through to the
design matrix and export.

**Acceptance criteria:**

- Config gains a generic, documented way to declare a subset of inputs as
  categorical/block predictors (e.g. `categorical_inputs: list[str]` with
  optional level metadata), validated on load with clear errors for unknown
  names or malformed levels.
- The declaration is machine-readable and round-trips through the existing
  config load/save path without breaking existing configs (default = empty
  list, so current behavior is unchanged when unset).
- No case-study literals; names are supplied by config only.
- `test_P0_S01_*` covers: valid declaration parses; unknown name raises;
  empty/default preserves current behavior.

**Fixture requirements:**

- Minimal in-memory/synthetic config dicts (no external files).

### Slice P0-S02: Design matrix includes categorical main effects + optional interactions (F1)

**Phase:** P0
**Depends on:** P0-S01
**Estimated size:** medium
**Files to create/modify:**

- src/rfm_pipeline/features.py
- tests/alignment/test_P0_S02_design_matrix_categorical.py

**Context.** With categorical inputs declared (P0-S01), the design-matrix
construction must emit categorical main-effect columns and support configurable
categorical×scalar interaction terms, so the fitted/exported model can depend on
them.

**Acceptance criteria:**

- Design-matrix construction encodes declared categorical inputs as
  main-effect column(s) and can add configured categorical×scalar interaction
  columns.
- Encoding is generic (works for arbitrary declared categoricals/levels) and
  deterministic; column names are stable and introspectable.
- When no categoricals are declared, output is byte-for-byte identical to
  current behavior (regression-guarded in the test).
- `test_P0_S02_*` covers: categorical main-effect columns appear; requested
  interaction columns appear; empty-declaration parity with baseline.

**Fixture requirements:**

- Small synthetic X with one scalar and one binary block column.

### Slice P0-S03: Exported evaluator honors categorical inputs — scenario-contrast test (F1)

**Phase:** P0
**Depends on:** P0-S02
**Estimated size:** medium
**Files to create/modify:**

- src/rfm_pipeline/final_ols.py
- tests/alignment/test_P0_S03_scenario_contrast.py

**Context.** F1 closure criterion: flipping a scenario switch can change the
prediction while all scalar inputs are held fixed. The exported coefficient
bundle + evaluator must round-trip categorical predictors.

**Acceptance criteria:**

- The exported model/evaluator accepts records containing the declared
  categorical inputs and applies their coefficients.
- Scenario-contrast test: for a fitted model with a non-zero categorical
  main effect, changing only the categorical value changes the prediction;
  with a zero categorical effect, predictions are invariant.
- Export artifact records the categorical columns/levels used.
- `test_P0_S03_*` implements the scenario-contrast assertion on synthetic
  data.

**Fixture requirements:**

- Synthetic fit where the block column has a known non-zero coefficient.

### Slice P0-S04: Sealed train/validation/test split protocol (F2)

**Phase:** P0
**Depends on:** none
**Estimated size:** medium
**Files to create/modify:**

- src/rfm_pipeline/data.py
- tests/alignment/test_P0_S04_sealed_test.py

**Context.** F2: the "untouched" holdout was consulted during development to pick
the pruning threshold (adaptive test-set reuse). Root cause: no protocol object
enforces that the final test partition is sealed during model development.

**Acceptance criteria:**

- A generic split protocol produces train / internal-validation / sealed-test
  partitions with configurable stratification keys (e.g. scenario strata).
- The sealed-test partition exposes an access guard: any read intended for
  model selection raises unless explicitly unsealed for the single final
  evaluation, and unsealing is recorded.
- Stratification balance is preserved and introspectable.
- `test_P0_S04_*` covers: strata balance; touching sealed test during
  selection raises; explicit final unseal succeeds and is logged.

**Fixture requirements:**

- Synthetic dataset with a stratification key column.

### Slice P0-S05: Threshold/pruning selection uses internal validation only (F2)

**Phase:** P0
**Depends on:** P0-S04
**Estimated size:** medium
**Files to create/modify:**

- src/rfm_pipeline/manuscript_pipeline_helpers.py
- tests/alignment/test_P0_S05_internal_selection.py

**Context.** F2/F6: the pruning threshold (`delta_threshold_override`) was chosen
against holdout nRMSE. Selection of any threshold/hyperparameter must use only
train + internal/nested validation, never the sealed test.

**Acceptance criteria:**

- Threshold/hyperparameter selection routines take an internal-validation
  scorer and never receive or read the sealed-test partition (enforced via the
  P0-S04 guard).
- The selected threshold is a function of internal-validation performance and
  is reproducible from a seed.
- `test_P0_S05_*` covers: selection runs using only train/val; attempting to
  pass the sealed test into selection raises; selection is deterministic.

**Fixture requirements:**

- Synthetic data + a trivial scorer with a known optimum threshold.

### Slice P0-S06: Frozen-config provenance stamp precedes test evaluation (F2)

**Phase:** P0
**Depends on:** P0-S05
**Estimated size:** small
**Files to create/modify:**

- src/rfm_pipeline/artifacts.py
- tests/alignment/test_P0_S06_frozen_provenance.py

**Context.** F2 closure: a time-stamped frozen configuration must predate the
final test predictions, with an auditable provenance record.

**Acceptance criteria:**

- A provenance API freezes the full resolved config (hash + timestamp) and
  records it before any sealed-test evaluation is permitted.
- Final-test evaluation refuses to run unless a matching frozen-config stamp
  exists with an earlier timestamp; the stamp hash matches the config used.
- `test_P0_S06_*` covers: freeze-then-evaluate succeeds; evaluate-without-freeze
  raises; config drift after freeze is detected.

**Fixture requirements:**

- Synthetic resolved-config object.

### Slice P0-S07: Interaction permutation-adequacy guard (F5)

**Phase:** P0
**Depends on:** none
**Estimated size:** medium
**Files to create/modify:**

- src/rfm_pipeline/manuscript_stages.py
- tests/alignment/test_P0_S07_perm_adequacy.py

**Context.** F5: 30 null permutations cannot resolve a 0.995 per-pair tail
(smallest corrected p = 1/31 ≈ 0.032). The interaction discovery stage must
validate that the permutation count is adequate for the requested tail
quantile / target error rate and fail loudly otherwise.

**Acceptance criteria:**

- Interaction discovery computes the minimum permutation count required for
  the configured tail/quantile and raises a clear error (or refuses to claim
  that tail) when the configured count is insufficient.
- The check is generic (function of requested quantile and family size).
- `test_P0_S07_*` covers: insufficient count for a 0.995 tail raises;
  sufficient count passes; boundary case.

**Fixture requirements:**

- No data needed; pure numeric adequacy check on synthetic parameters.

### Slice P0-S08: Multiplicity control across interaction pairs (F5)

**Phase:** P0
**Depends on:** P0-S07
**Estimated size:** large
**Files to create/modify:**

- src/rfm_pipeline/manuscript_stages.py
- tests/alignment/test_P0_S08_interaction_multiplicity.py

**Context.** F5: the per-pair 0.5% rule controls no family-wise/false-discovery
error across the 2,346 candidate pairs. Implement shared-response permutations
with a max-statistic/step-down FWER rule, or corrected p-values with a defensible
FDR procedure.

**Acceptance criteria:**

- A generic multiplicity-controlled selection rule (FWER max-stat/step-down or
  BH-FDR) operates over the full candidate family using shared-response
  permutations.
- Under a complete null (synthetic, all pairs null), the expected number of
  false selections is controlled at the configured level (validated
  statistically with a fixed seed and tolerance).
- Under a planted-signal alternative, true interactions are recovered (power
  sanity check).
- `test_P0_S08_*` covers: null false-selection control; planted-signal
  recovery.

**Fixture requirements:**

- Synthetic response with a known interaction structure and a pure-null
  variant; fixed RNG seed.

### Slice P0-S09: Correct `empirical_null_retained` provenance semantics (F5)

**Phase:** P0
**Depends on:** none
**Estimated size:** small
**Files to create/modify:**

- src/rfm_pipeline/manuscript_stages.py
- tests/alignment/test_P0_S09_provenance_retained.py

**Context.** F5 bug: `empirical_null_retained` is populated with every candidate
name rather than the actually-retained set, making the diagnostic wrong.

**Acceptance criteria:**

- `empirical_null_retained` (and the paired count fields) reflect only the
  retained set; the count equals the number of retained candidates.
- Applies consistently to both interaction and nonlinear provenance blocks.
- `test_P0_S09_*` covers: retained flag/count match the true retained subset
  on a synthetic scoring frame; non-retained candidates are flagged False.

**Fixture requirements:**

- Synthetic candidate-score frame with a known retained subset.

### Slice P0-S10: Nonlinear discovery multiplicity + discovery/choice separation (F5)

**Phase:** P0
**Depends on:** P0-S09
**Estimated size:** large
**Files to create/modify:**

- src/rfm_pipeline/manuscript_stages.py
- tests/alignment/test_P0_S10_nonlinear_multiplicity.py

**Context.** F5: nonlinear discovery tests 276 transforms × 20 components,
selecting the best transform without multiplicity correction or independent
validation, and accounting for search across inputs/components/transform
families.

**Acceptance criteria:**

- Nonlinear discovery applies a documented multiplicity correction accounting
  for the full search (inputs × components × transforms), and separates
  discovery from transform choice via nested/independent validation.
- Under a complete null, false-selection is controlled at the configured level;
  under a planted nonlinearity, the correct transform family is recovered.
- `test_P0_S10_*` covers: null control; planted-nonlinearity recovery.

**Fixture requirements:**

- Synthetic component responses with one planted nonlinear transform + a null
  variant; fixed seed.

### Slice P0-S11: Validated support-selection rule via internal validation (F6)

**Phase:** P0
**Depends on:** P0-S05, P0-S08
**Estimated size:** large
**Files to create/modify:**

- src/rfm_pipeline/final_ols.py
- src/rfm_pipeline/regularized_screening.py
- tests/alignment/test_P0_S11_support_recovery.py

**Context.** F6: LASSO/stability/HC3 stages retained all enriched terms; only the
no-refit marginal-impact threshold reduced the set, and that threshold was
holdout-tuned. Replace the no-refit removal approximation with a justified
selection rule whose threshold is chosen by internal validation, and demonstrate
actual support reduction on synthetic data with a known sparse truth.

**Acceptance criteria:**

- A generic support-selection rule reduces an enriched candidate set to a
  sparse support using a refit-based criterion with the threshold selected on
  internal validation only (reusing P0-S05 machinery; never the sealed test).
- On synthetic data with a known sparse generative support, the rule recovers
  (approximately) the true support: it removes null terms and retains signal
  terms above a documented tolerance.
- Reports threshold-sensitivity of the resulting support using internal
  validation.
- `test_P0_S11_*` covers: null-term removal; signal-term retention;
  internal-only threshold selection.

**Fixture requirements:**

- Synthetic design + response with a known sparse coefficient vector; fixed
  seed.

### Slice P0-S12: Per-stage support-composition provenance (F6)

**Phase:** P0
**Depends on:** P0-S11
**Estimated size:** small
**Files to create/modify:**

- src/rfm_pipeline/manuscript_stages.py
- tests/alignment/test_P0_S12_stage_support_provenance.py

**Context.** F6 closure: the workflow must report honestly which stage changed
the support (so the manuscript can no longer claim sparsification by stages that
were no-ops).

**Acceptance criteria:**

- The pipeline emits a per-stage support-composition record: candidate count
  in, count out, and the set actually removed by each stage (screen, LASSO,
  stability, HC3, final selection).
- The record makes "this stage removed 0 terms" explicit and machine-checkable.
- `test_P0_S12_*` covers: a synthetic run where one stage removes terms and
  another removes none yields correct per-stage in/out/removed sets.

**Fixture requirements:**

- Synthetic multi-stage selection trace.

### Slice P0-S13: Remove case-study-specific modules from generic src (F4 / repo hygiene)

**Phase:** P0
**Depends on:** none
**Estimated size:** medium
**Files to create/modify:**

- src/rfm_pipeline/track_a_v3.py
- src/rfm_pipeline/wave5_track_a.py
- src/rfm_pipeline/__init__.py
- tests/alignment/test_P0_S13_no_casestudy_code.py

**Context.** F3/F4: the planning add-on and wave-mixing sensitivity code are
case-study-specific artifacts that were removed from the manuscript. They also
violate the "100% generic" rule by living in generic `src/`. Remove them (and
any now-dangling exports/tests) so the workflow matches the manuscript and the
repo stays case-study-agnostic.

**Acceptance criteria:**

- `src/rfm_pipeline/track_a_v3.py` and `src/rfm_pipeline/wave5_track_a.py` are
  removed, along with their exports from `__init__.py` and any dedicated tests
  that only exercised them (`tests/test_track_a_v3.py`,
  `tests/test_wave5_track_a.py`).
- The remaining `src/` contains no case-study literals for the removed
  add-on; `import rfm_pipeline` still succeeds.
- `test_P0_S13_*` covers: importing the removed modules raises ImportError;
  `rfm_pipeline` top-level import still works; the removed symbols are absent
  from `rfm_pipeline.__all__`.
- Broader guard: `pixi run python -m pytest -q tests/test_import_smoke.py`
  still passes.

**Fixture requirements:**

- None.

______________________________________________________________________

## PHASE P1 — evaluation completeness

### Slice P1-S01: Stratified bootstrap with adequate replicates (M3)

**Phase:** P1
**Depends on:** P0-S04
**Estimated size:** medium
**Files to create/modify:**

- src/rfm_pipeline/metrics.py
- tests/alignment/test_P1_S01_stratified_bootstrap.py

**Context.** M3: 100 percentile-bootstrap resamples cannot justify four-decimal
endpoints and were not stratified. Support ≥1000 (configurable) stratified
resamples with within-stratum resampling, endpoint Monte-Carlo stability
diagnostics, predefined rounding, and paired bootstrap differences.

**Acceptance criteria:**

- `bootstrap_macro_nrmse_ci` (or a companion) supports configurable replicate
  count (default ≥1000), within-stratum resampling given a strata key, and
  returns Monte-Carlo endpoint-stability diagnostics.
- A paired-bootstrap difference helper for model comparison exists.
- `test_P1_S01_*` covers: stratified resampling stays within strata;
  increasing replicates shrinks endpoint MC variability; paired-difference
  interval on synthetic data.

**Fixture requirements:**

- Synthetic per-output errors with a strata key; fixed seed.

### Slice P1-S02: Stratified + excluded-output error reporting (M4)

**Phase:** P1
**Depends on:** none
**Estimated size:** medium
**Files to create/modify:**

- src/rfm_pipeline/metrics.py
- tests/alignment/test_P1_S02_stratified_metrics.py

**Context.** M4: headline macro nRMSE is computed on the conditioned output
subset; the excluded outputs and stratified/tail behavior are not reported.

**Acceptance criteria:**

- Metrics helpers report error stratified by arbitrary keys (scenario, module,
  year), plus median, upper-tail (e.g. p95/p99), worst-output, and a separate
  absolute/domain-scaled error summary for the excluded-output set.
- Every headline-metric result carries an explicit metric-population label
  (which outputs it covers).
- `test_P1_S02_*` covers: stratified breakdown correctness; excluded-output
  summary; population label present.

**Fixture requirements:**

- Synthetic outputs with strata keys and an included/excluded mask.

### Slice P1-S03: Competitive baseline model interface (M5)

**Phase:** P1
**Depends on:** P0-S04
**Estimated size:** large
**Files to create/modify:**

- src/rfm_pipeline/baselines.py
- tests/alignment/test_P1_S03_baselines.py

**Context.** M5: no alternative surrogate class is benchmarked, a JDS desk-review
risk. Provide a generic baseline interface and comparison harness on the same
frozen split and metric.

**Acceptance criteria:**

- A generic `fit`/`predict` baseline interface with at least: reduced-rank or
  ridge/PLS, a sparse-linear (elastic-net) baseline, and per-stratum
  first-order models.
- A comparison harness records accuracy, model size, fit time, evaluation
  time, and peak memory on the same split.
- Baselines are case-study-agnostic (operate on arbitrary X/Y).
- `test_P1_S03_*` covers: each baseline fits/predicts on synthetic data; the
  harness returns the full comparison record schema.

**Fixture requirements:**

- Small synthetic multi-output regression dataset.

### Slice P1-S04: Synthetic stress-test harness for blind spots (M6)

**Phase:** P1
**Depends on:** P0-S08, P0-S11
**Estimated size:** large
**Files to create/modify:**

- src/rfm_pipeline/stress_tests.py
- tests/alignment/test_P1_S04_stress_tests.py

**Context.** M6: pure interactions, symmetric nonlinearities, rare output-specific
signals, and discarded PCA variance are central failure modes. Provide a generic
stress-test harness that plants each structure and reports false exclusions and
downstream error.

**Acceptance criteria:**

- A generic harness generates synthetic DGPs for: pure interaction, symmetric
  nonlinearity, rare-output signal, PCA-threshold sensitivity, and
  range-threshold sensitivity.
- For each, it runs the relevant screening/discovery stage and reports false
  exclusions and downstream error, not just aggregate nRMSE.
- The harness is parameterized (no case-study constants).
- `test_P1_S04_*` covers: each DGP generator produces the intended structure;
  the harness reports false-exclusion metrics; a planted signal is detectable
  at high strength and missed at zero strength.

**Fixture requirements:**

- Deterministic synthetic DGP generators; fixed seeds.

### Slice P1-S05: Manuscript-data-contract claim↔artifact check (F3)

**Phase:** P1
**Depends on:** none
**Estimated size:** medium
**Files to create/modify:**

- src/rfm_pipeline/manuscript_data_contract.py
- tests/alignment/test_P1_S05_data_contract.py

**Context.** F3 closure: every hard-coded manuscript number/figure should map to a
source artifact, with an automated check that fails when a claim disagrees with
its source table.

**Acceptance criteria:**

- A generic contract utility maps declared claims (label → expected value →
  source artifact path/column) and validates each against the source, with a
  tolerance for floats and a clear diff on mismatch.
- The utility is data-driven (claims supplied by a manifest, not hardcoded).
- `test_P1_S05_*` covers: matching claim passes; mismatched claim fails with a
  useful message; missing source raises.

**Fixture requirements:**

- Synthetic claims manifest + a tiny synthetic source table.

______________________________________________________________________

## PHASE P2 — reproducibility tooling

### Slice P2-S01: One-command release/reproduction manifest with checksums (F7)

**Phase:** P2
**Depends on:** P0-S06
**Estimated size:** medium
**Files to create/modify:**

- src/rfm_pipeline/release_manifest.py
- scripts/build_release_manifest.py
- tests/alignment/test_P2_S01_release_manifest.py

**Context.** F7: reproducibility resources are promises. Provide generic tooling
that enumerates the artifacts needed to reproduce reported tables/figures, emits
checksums and an environment/lockfile reference, and produces a one-command
manifest usable from a clean checkout.

**Acceptance criteria:**

- A generic release-manifest builder walks a declared artifact set, records
  SHA-256 checksums, sizes, and a captured environment/lockfile reference, and
  writes a machine-readable manifest.
- A verify mode recomputes checksums and reports drift.
- The CLI script wraps the builder for one-command use via Pixi.
- `test_P2_S01_*` covers: manifest build over a synthetic artifact tree;
  checksum verification detects a tampered file; environment reference present.

**Fixture requirements:**

- Temporary synthetic artifact directory.

______________________________________________________________________

## Downstream milestone (not slice-runner slices)

These require HPC, data rights, LaTeX edits, or release infrastructure and are
tracked in `docs/AGENT_SYNC.md` / `docs/scope_backlog.md`:

- Rerun the frozen case-study pipeline end-to-end with the corrected workflow
  (scenario predictors, sealed test, validated selection) and regenerate every
  artifact and manuscript number (F1/F2/F5/F6 closure on real data).
- Sensitivity-study re-execution from one locked manifest with attrition and
  DGP-level uncertainty (M1/M2/M7).
- AEO/BSM data-provenance capture and redistribution licensing (M8).
- Public anonymized code/data/supplement releases + DOIs (F7 releases).
- Manuscript prose: separate prediction/interpretation/inference claims (M9);
  rewrite abstract/cover/results/discussion/model-card from regenerated
  artifacts; JDS admin (ORCID/ScholarOne).
