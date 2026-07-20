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

______________________________________________________________________

## PHASE R2 — round-1 regression remediation

### Slice R2-S01: Reconcile permutation-adequacy guard with structural discovery tests (F5 regression)

**Phase:** R2
**Depends on:** none
**Estimated size:** medium
**Files to create/modify:**

- src/rfm_pipeline/manuscript_stages.py
- tests/test_manuscript_interaction_discovery.py
- tests/alignment/test_R2_S01_guard_reconciliation.py

**Context.** The P0-S07 permutation-adequacy guard (`check_permutation_adequacy`,
called from interaction discovery) now raises `PermutationAdequacyError` for
8 pre-existing structural tests in
`tests/test_manuscript_interaction_discovery.py`
(`test_interaction_discovery_generates_all_pairs_from_retained_first_order_terms`,
`test_interaction_discovery_respects_candidate_pair_range`,
`test_interaction_discovery_rejects_empty_candidate_pair_range`,
`test_interaction_discovery_rejects_invalid_parallel_backend`,
`test_interaction_discovery_uses_configured_parallel_backend_without_fallback`,
`test_interaction_discovery_accepts_dask_backend_with_executor`,
`test_interaction_discovery_falls_back_to_joblib_when_dask_executor_fails`,
`test_interaction_discovery_resumes_from_checkpointed_permutation_scores`).
These tests deliberately use tiny `permutation_count_B` (2, 3, 19) with **mocked**
scoring to exercise pair generation, parallel-backend selection, and
checkpoint/resume plumbing — they do not validate statistical tail behavior.
The guard must keep protecting real analysis while these plumbing tests still run.

**Requirements (do NOT weaken the guard's real-analysis default and do NOT weaken
any test's actual assertion target):**

- Add an explicit `enforce_permutation_adequacy: bool = True` field to
  `InteractionDiscoverySpec` (default True → real runs remain guarded; F5
  behavior unchanged when unset). The guard runs only when True.
- Update the 8 structural tests to construct their spec with
  `enforce_permutation_adequacy=False`, since they intentionally use tiny
  mocked-scoring configs and assert plumbing, not tail validity. Do not change
  any other assertion in those tests. Do not change their `permutation_count_B`
  values (checkpoint/backend assertions depend on them).
- `min_permutations_required` / `check_permutation_adequacy` behavior is
  unchanged; the config default still enforces.

**Acceptance criteria:**

- `test_R2_S01_*` asserts: with `enforce_permutation_adequacy=True` (default) an
  inadequate `B` for the configured quantile still raises
  `PermutationAdequacyError`; with `enforce_permutation_adequacy=False` the same
  inadequate `B` does not raise from the guard; the default value of the new
  field is `True`.
- The sub-agent MUST also confirm the previously-failing suite is green:
  `pixi run python -m pytest -q tests/test_manuscript_interaction_discovery.py`
  exits 0.
- `pixi run ruff check src/rfm_pipeline/manuscript_stages.py tests/test_manuscript_interaction_discovery.py tests/alignment/test_R2_S01_guard_reconciliation.py` is clean.

**Fixture requirements:**

- Reuse existing InteractionDiscoverySpec construction patterns; synthetic only.

______________________________________________________________________

## PHASE R3 — independent-review remediation (b96442b..e31fb15 review)

An independent code review found the corrected F5 methodology was implemented and
unit-tested but NOT wired into the production discovery path (the workflow still
ran the uncorrected rule), plus real correctness bugs. These slices close that gap.

### Slice R3-S01: Wire multiplicity-controlled interaction selection into the production path (F5 BLOCKING)

**Phase:** R3
**Depends on:** none
**Estimated size:** large
**Files to create/modify:**

- src/rfm_pipeline/manuscript_stages.py
- tests/test_manuscript_interaction_discovery.py
- tests/alignment/test_R3_S01_interaction_wiring.py

**Context.** `multiplicity_controlled_interaction_selection` (FWER max-stat /
BH-FDR) exists and is unit-tested (P0-S08) but is called ONLY by its test. The
production function `discover_manuscript_interactions` still retains pairs with the
uncorrected per-pair rule at manuscript_stages.py:2529
(`retained = observed_scores > thresholds`, a per-pair `null_threshold_quantile`
cut). The F5 defect the slice claimed to fix is therefore still shipped. Also, the
permutation-adequacy guard is currently invoked with `family_size=total pairs`
while the retention is per-pair, so the guard is statistically inconsistent with
the inference performed.

**Requirements (do NOT weaken tests; do NOT loosen assertions to pass):**

- Route production retention in `discover_manuscript_interactions` through
  `multiplicity_controlled_interaction_selection(observed_scores, null_statistics, alpha=..., method=...)` and use its `selected` mask; expose
  the family-error method + level via `InteractionDiscoverySpec` (documented
  default). The corrected family-wise/FDR selection becomes the default.
- Make the adequacy guard's `family_size` consistent with the inference actually
  performed now that selection is family-corrected (i.e. the guard budget must
  match the corrected family-wise procedure, not the removed per-pair rule).
- Existing integration tests in `tests/test_manuscript_interaction_discovery.py`
  that asserted specific retained pairs under the OLD per-pair rule must be
  updated to the CORRECT expected outputs of the corrected rule, with the
  expected values derived from the corrected procedure — not by loosening the
  assertion. Add at least one guard-ON integration test that exercises the real
  production path (do not leave the guard-ON path untested).

**Acceptance criteria:**

- `test_R3_S01_*` (in tests/alignment) calls the PRODUCTION interaction
  discovery entry point (`discover_manuscript_interactions` /
  `run_interaction_discovery_stage`) and asserts: under a pure null, the
  corrected rule yields far fewer false selections than an uncorrected per-pair
  0.5% cut on the same statistics (family error is actually controlled); under a
  planted strong interaction, that interaction is retained.
- `grep` shows `multiplicity_controlled_interaction_selection` is now referenced
  in `src/rfm_pipeline/manuscript_stages.py` (wired in), not only in tests.
- The sub-agent MUST confirm green:
  `pixi run python -m pytest -q tests/test_manuscript_interaction_discovery.py tests/alignment/test_P0_S07_perm_adequacy.py tests/alignment/test_P0_S08_interaction_multiplicity.py`
- `pixi run ruff check src/rfm_pipeline/manuscript_stages.py tests/test_manuscript_interaction_discovery.py tests/alignment/test_R3_S01_interaction_wiring.py` is clean.

**Fixture requirements:**

- Synthetic observed + null interaction statistics with a pure-null variant and a
  planted-signal variant; fixed seed.

### Slice R3-S02: Wire nonlinear multiplicity correction + fix min-p tracking into production (F5 BLOCKING + bug)

**Phase:** R3
**Depends on:** none
**Estimated size:** large
**Files to create/modify:**

- src/rfm_pipeline/manuscript_stages.py
- tests/alignment/test_R3_S02_nonlinear_wiring.py

**Context.** `nonlinear_discovery_with_multiplicity_correction` (Bonferroni
`corrected_alpha`, `choose_transform_by_cv`) exists/tested (P0-S10) but no
production runner calls it. Production `discover_manuscript_nonlinear_transformations`
→ `_score_one_nonlinear_feature` (manuscript_stages.py ~6352-6376) uses a RAW
uncorrected per-(feature,component) p-threshold and chooses the transform by
minimum RMSE against the GAM smooth on the SAME training data (no independent CV).
Additionally there is a min-p tracking bug: `best_edf`/`best_p` advance jointly
under a disjunctive condition (~6352-6356 and ~3327-3331), so `best_p` is not the
minimum p-value across components as documented.

**Requirements (no test weakening):**

- Route the production nonlinear stage through the corrected multiplicity control
  (apply `corrected_alpha` accounting for inputs × components × transforms) and
  separate discovery from transform choice via independent/nested CV
  (`choose_transform_by_cv`). The corrected procedure becomes the default.
- Fix min-p tracking so the significance decision uses the true minimum p-value
  across components (`best_p = min(best_p, p)` tracked independently of the
  EDF-based best-component selection).

**Acceptance criteria:**

- `test_R3_S02_*` calls the PRODUCTION nonlinear discovery entry point and
  asserts: under a pure null, false selections are controlled at the configured
  level (materially fewer than the uncorrected per-component threshold); under a
  planted nonlinear transform, the correct transform family is recovered; and the
  reported min p-value equals the true minimum across components on a crafted case.
- `grep` shows the corrected nonlinear function/`corrected_alpha`/
  `choose_transform_by_cv` are referenced by the production path in
  `src/rfm_pipeline/manuscript_stages.py`.
- Confirm green: `pixi run python -m pytest -q tests/alignment/test_P0_S10_nonlinear_multiplicity.py`
  and any existing nonlinear-discovery tests.
- `pixi run ruff check src/rfm_pipeline/manuscript_stages.py tests/alignment/test_R3_S02_nonlinear_wiring.py` is clean.

**Fixture requirements:**

- Synthetic component responses: pure-null variant, planted-transform variant, and
  a crafted multi-component case for the min-p assertion; fixed seed.

### Slice R3-S03: Make the no-case-study-code invariant test honest (F4 test integrity)

**Phase:** R3
**Depends on:** none
**Estimated size:** small
**Files to create/modify:**

- tests/alignment/test_P0_S13_no_casestudy_code.py

**Context.** `test_P0_S13_no_casestudy_code.py` claims to guarantee `src/` stays
case-study-agnostic but only checks that two removed modules fail to import. It does
NOT scan `src/` for case-study literals, and pre-existing case-study-specific
scenario helpers remain in `src/rfm_pipeline/data.py` (`AFSC`/`UAEORO` default column
names, `add_scenario_flags`) and provenance docstrings in `config.py`. Full
generalization of those helpers is a separate, larger milestone (tracked in
`docs/scope_backlog.md`); this slice makes the test HONEST so it prevents NEW
leakage and documents the known exceptions.

**Requirements (do not weaken; do not delete the existing import/`__all__` checks):**

- Add a filesystem scan over `src/rfm_pipeline/**/*.py` for a denylist of
  case-study tokens (e.g. `track_a_v3`, `wave5`, `wave1234`, and the removed
  planning-add-on symbols) that MUST be absent, so re-introducing the removed
  case-study code fails the test.
- Explicitly document (in the test) the known pre-existing scenario-helper
  exceptions (`AFSC`/`UAEORO`/`add_scenario_flags`) as a tracked
  generalization-backlog item rather than silently ignoring them, so the test's
  claim matches reality.

**Acceptance criteria:**

- `test_P0_S13_*` fails if any denylisted removed-case-study token reappears in
  `src/rfm_pipeline/`, and passes on the current tree.
- Existing import-absence and `__all__`-absence assertions are retained.
- `pixi run ruff check tests/alignment/test_P0_S13_no_casestudy_code.py` is clean.

**Fixture requirements:**

- None (scans the real src tree).

### Slice R3-S04: Persist resolved categorical levels at fit for reproducible prediction (F1 correctness)

**Phase:** R3
**Depends on:** none
**Estimated size:** medium
**Files to create/modify:**

- src/rfm_pipeline/features.py
- src/rfm_pipeline/final_ols.py
- tests/alignment/test_R3_S04_categorical_levels_persist.py

**Context.** When a `CategoricalInputDecl` has `levels=None`, levels are inferred
from training data at fit time but the stored `DesignMatrixSpec` keeps `levels=None`.
At predict time levels are re-inferred from the passed records, so a small
scenario-contrast batch produces a different (often empty after `drop_first`)
indicator set and `predict_final_ols` raises "Missing retained feature columns".
The F1 "flip a categorical → prediction changes" round-trip only works when explicit
levels are declared.

**Requirements:**

- Persist the resolved levels into the stored `DesignMatrixSpec` at fit time so
  encoding at predict time is reproducible regardless of the prediction batch and
  independent of `drop_first` on small batches.

**Acceptance criteria:**

- `test_R3_S04_*` fits with `levels=None` on training data, then predicts on a
  2-row scenario-contrast batch and asserts: no "missing columns" error, and
  flipping the categorical changes the prediction when its coefficient is nonzero.
- `pixi run ruff check src/rfm_pipeline/features.py src/rfm_pipeline/final_ols.py tests/alignment/test_R3_S04_categorical_levels_persist.py` is clean.

**Fixture requirements:**

- Synthetic fit with `levels=None` and a nonzero categorical effect.

### Slice R3-S05: Stop fabricating empirical provenance on the ElasticNet interaction path (provenance integrity)

**Phase:** R3
**Depends on:** none
**Estimated size:** small
**Files to create/modify:**

- src/rfm_pipeline/manuscript_stages.py
- tests/alignment/test_R3_S05_elasticnet_provenance.py

**Context.** The non-default `elasticnet_interactions` method emits
`p_value = 0.0/1.0`, `null_threshold=0.0`, and an all-zero
`interaction_null_summary` (manuscript_stages.py ~2024-2062). These are not real
empirical p-values/null statistics; downstream multiplicity/provenance consumers
would be misled into treating them as significant permutation results.

**Requirements:**

- On the ElasticNet path, emit `NaN`/`None` for null-derived fields (p-value,
  null threshold, null summary) rather than fabricated zeros, so consumers can
  distinguish "not computed" from "significant".

**Acceptance criteria:**

- `test_R3_S05_*` asserts the ElasticNet interaction path yields NaN/None (not
  0.0/1.0) for null-derived provenance fields, while still reporting its selection.
- `pixi run ruff check src/rfm_pipeline/manuscript_stages.py tests/alignment/test_R3_S05_elasticnet_provenance.py` is clean.

**Fixture requirements:**

- Synthetic interaction inputs exercising the `elasticnet_interactions` method.

## PHASE G — Generalize scenario/categorical handling (remove BSM AFSC/UAEORO from generic src)

Goal: the generic pipeline must not encode any case study's scenario scheme.
Replace the hardcoded two-binary `AFSC`/`UAEORO` mechanism with generic,
parameter-driven categorical handling and remove all AFSC/UAEORO literals + the
BSM composite-label parser. See docs/decision_log.md (2026-07-18) for the design.

Cross-cutting rules for every G1 slice:

- No new case-study literals. Keep behavior deterministic.
- The per-slice validator only runs `tests/alignment/`; you MUST also update the
  existing tests named in each slice so the FULL suite stays green — do not delete
  or weaken them, migrate their assertions to the generic API with correct expected
  values.
- Run `pixi run ruff check <changed files>` (clean) as part of each slice.

### Slice G1-S01: Generic combination labels + stratified subset in data.py

**Phase:** G1
**Depends on:** none
**Estimated size:** large
**Files to create/modify:**

- src/rfm_pipeline/data.py
- src/rfm_pipeline/__init__.py
- tests/test_data.py
- tests/alignment/test_G1_S01_generic_combination.py

**Context.** `add_scenario_flags` + `_parse_on_off_flag` parse the BSM composite
label `AFSC{0|1}_UAEORO{0|1}` at fixed character offsets — case-study data ingest
that must not live in the generic pipeline. `make_boolean_combination_labels` and
`stratified_subset_by_boolean_combination` hardcode the two columns AFSC/UAEORO and
the four expected combinations.

**Requirements:**

- REMOVE `add_scenario_flags` and `_parse_on_off_flag` entirely (and their
  `__init__.py` import + `__all__` entry for `add_scenario_flags`).
- Rename `make_boolean_combination_labels` ->
  `combination_labels(frame, *, columns: Sequence[str], output_column: str = "combination") -> pd.Series`
  producing deterministic labels by joining `f"{col}={value}"` for each column in
  the given order with `"|"`. Update the `__init__.py` export accordingly.
- Rename `stratified_subset_by_boolean_combination` ->
  `stratified_subset_by_combination(frame, *, columns: Sequence[str], n_per_combination: int = 5000, random_state: int = 123, require_all_combinations: bool = True) -> pd.DataFrame`.
  Drop the hardcoded four-element `expected` list: the strata are the observed
  combinations (sorted deterministically); when `require_all_combinations` is True,
  every observed combination must have >= `n_per_combination` rows. Update the
  `__init__.py` export accordingly.
- Preserve deterministic RNG behavior (per-combination `rng.choice`, sorted stratum
  order).

**Acceptance criteria:**

- `test_G1_S01_*` builds a frame with 3 categorical columns (not two, not named
  AFSC/UAEORO) and asserts: `combination_labels` yields the expected `col=val|...`
  labels; `stratified_subset_by_combination` returns `n_per_combination` rows per
  observed combination, is deterministic under a fixed seed, and raises when a
  combination is too small.
- No AFSC/UAEORO/`add_scenario_flags` token remains in data.py.
- `tests/test_data.py` is migrated to the generic API (no calls to removed
  functions; combination/subset assertions use generic column names) and passes.
- `pixi run ruff check src/rfm_pipeline/data.py src/rfm_pipeline/__init__.py tests/test_data.py tests/alignment/test_G1_S01_generic_combination.py` is clean.

**Fixture requirements:**

- Synthetic frame with >=3 categorical columns and enough rows per combination.

### Slice G1-S02: Generic stratification columns in stratified_holdout_split

**Phase:** G1
**Depends on:** G1-S01
**Estimated size:** medium
**Files to create/modify:**

- src/rfm_pipeline/data.py
- tests/test_data.py
- tests/alignment/test_G1_S02_generic_holdout.py

**Context.** `stratified_holdout_split` hardcodes `afsc_column="AFSC"`,
`uaeoro_column="UAEORO"` and builds the stratify label from those two columns.

**Requirements:**

- Replace the `afsc_column`/`uaeoro_column` parameters with
  `stratify_columns: Sequence[str] = ()`. Build the stratify label by joining
  `astype(str)` of each column in `stratify_columns` with `"_"` when all are present
  in `X_aligned`; otherwise fall back to `scenario_column` (column or MultiIndex
  level) exactly as today. Preserve the single-unique-value guard and the
  `train_test_split` determinism.

**Acceptance criteria:**

- `test_G1_S02_*` asserts stratification on an arbitrary list of columns (generic
  names) reproduces the expected deterministic split and preserves stratum
  proportions; and that the `scenario_column` fallback still works when
  `stratify_columns` is empty/absent.
- No AFSC/UAEORO token remains in `stratified_holdout_split`.
- `tests/test_data.py` holdout tests migrated to `stratify_columns=` and pass.
- `pixi run ruff check src/rfm_pipeline/data.py tests/test_data.py tests/alignment/test_G1_S02_generic_holdout.py` is clean.

**Fixture requirements:**

- Synthetic X/Y with generic categorical stratification columns.

### Slice G1-S03: Remove AFSC/UAEORO literal special-cases in features.py and manuscript_stages.py

**Phase:** G1
**Depends on:** none
**Estimated size:** small
**Files to create/modify:**

- src/rfm_pipeline/features.py
- src/rfm_pipeline/manuscript_stages.py
- tests/test_features.py
- tests/alignment/test_G1_S03_no_scenario_specialcase.py

**Context.** `canonical_module_from_factor_name` special-cases
`name in {"AFSC","UAEORO"}` -> `"Scenario"`. `_legacy_normalize_factor_token`
maps `"UAEORO"`/`"AFSC"` to BSM factor names.

**Requirements:**

- Remove the `{"AFSC","UAEORO"}` special-case from
  `canonical_module_from_factor_name` (dot-scoped -> prefix, else "Unscoped").
- Remove the `"UAEORO"`/`"AFSC"` branches from `_legacy_normalize_factor_token`
  (return the normalized/stripped token unchanged for those inputs). Do NOT touch
  the separate BSM fuel-pathway taxonomy in `_legacy_module_from_factor_name`
  (out of scope — backlogged).

**Acceptance criteria:**

- `test_G1_S03_*` asserts `canonical_module_from_factor_name` no longer returns
  "Scenario" for the literal strings "AFSC"/"UAEORO" (they are treated as generic
  unscoped names), and that scoped names ("X.y") still map to their prefix.
- No AFSC/UAEORO token remains in features.py or in `_legacy_normalize_factor_token`.
- `tests/test_features.py` migrated (drop assertions asserting the AFSC/UAEORO
  scenario special-case; keep/upgrade generic module-prefix assertions) and passes.
- `pixi run ruff check src/rfm_pipeline/features.py src/rfm_pipeline/manuscript_stages.py tests/test_features.py tests/alignment/test_G1_S03_no_scenario_specialcase.py` is clean.

**Fixture requirements:**

- None beyond literal factor-name inputs.

### Slice G1-S04: Generic demo manuscript fixtures (rename AFSC/UAEORO)

**Phase:** G1
**Depends on:** none
**Estimated size:** medium
**Files to create/modify:**

- src/rfm_pipeline/manuscript_runtime.py
- tests/test_pipeline_smoke.py
- tests/test_feature_expansion.py
- tests/alignment/test_G1_S04_generic_demo_fixtures.py

**Context.** `write_demo_manuscript_artifacts` builds a toy dataset whose two
boolean categorical columns are literally named `AFSC`/`UAEORO`, propagated into
input_metadata and feature_catalog and used in the y2/y3 formulas. The
manuscript-reproduction-smoke check consumes these artifacts.

**Requirements:**

- Rename the two boolean demo columns from `AFSC`/`UAEORO` to generic names
  `cat_a`/`cat_b` throughout `write_demo_manuscript_artifacts` (data frame,
  input_metadata `input_name`, feature_catalog `feature_name` entries and any
  derived interaction/feature names) and update the y2/y3 formula variable
  references accordingly.
- Keep the dataset deterministic and self-consistent (feature_catalog names must
  match producible design-matrix columns).

**Acceptance criteria:**

- `test_G1_S04_*` asserts the demo artifacts contain `cat_a`/`cat_b` (not
  AFSC/UAEORO) and remain internally consistent (metadata/catalog names align with
  the input matrix columns).
- `pixi run manuscript-reproduction-smoke` (or the equivalent pytest exercising it)
  passes with the renamed fixtures.
- `tests/test_pipeline_smoke.py` and `tests/test_feature_expansion.py` migrated to
  the generic names and pass.
- No AFSC/UAEORO token remains in manuscript_runtime.py.
- `pixi run ruff check src/rfm_pipeline/manuscript_runtime.py tests/test_pipeline_smoke.py tests/test_feature_expansion.py tests/alignment/test_G1_S04_generic_demo_fixtures.py` is clean.

**Fixture requirements:**

- Uses the in-repo demo-artifact writer.

### Slice G1-S05: Enforce AFSC/UAEORO absence + de-BSM config docstrings

**Phase:** G1
**Depends on:** G1-S01, G1-S02, G1-S03, G1-S04
**Estimated size:** small
**Files to create/modify:**

- src/rfm_pipeline/config.py
- tests/alignment/test_P0_S13_no_casestudy_code.py
- tests/alignment/test_G1_S05_enforce_scenario_generic.py

**Context.** After G1-S01..S04 no AFSC/UAEORO literal remains in src. The
P0-S13/R3-S03 denylist previously documented AFSC/UAEORO as pre-existing
exceptions; that exception is now obsolete. config.py docstrings tie defaults to
the BSM/JDS publication run.

**Requirements:**

- Remove `AFSC`/`UAEORO` from the documented exceptions in
  `test_P0_S13_no_casestudy_code.py` and ADD them to the enforced denylist tokens
  (the src scan must now fail if either reappears).
- Strip BSM/JDS-specific phrasing from config.py docstrings (make them generic),
  without changing any config field names, defaults, or behavior.

**Acceptance criteria:**

- `test_G1_S05_*` (or the updated P0-S13 scan) fails if `AFSC`/`UAEORO` appears
  anywhere under src/rfm_pipeline and passes on the current tree.
- No behavioral change to config schema/defaults (existing config tests still pass).
- `pixi run ruff check src/rfm_pipeline/config.py tests/alignment/test_P0_S13_no_casestudy_code.py tests/alignment/test_G1_S05_enforce_scenario_generic.py` is clean.

**Fixture requirements:**

- None (filesystem denylist scan of src/rfm_pipeline).

______________________________________________________________________

## Phase RS — Prespecified semi-synthetic recovery study (method-evidence)

Motivation: on the BSM fit, LASSO / stability / HC3 retain every enriched term,
so the case study alone does not establish general sparse-support recovery or
FWER control. Per `bsm-public-rf-manuscript/docs/ANALYSIS_HANDOFF.md`
("Method-evidence requirement"), add (a) an exact finite-permutation maxT
interaction-FWER rule with global-null validation, (b) a prespecified
known-support recovery study with recovery estimands and competitive
comparators, and (c) a small, fully-local reproduction run producing released
artifacts. All code stays 100% case-study-agnostic (no BSM constants).

### Slice RS-S01: Exact/conservative finite-permutation maxT interaction-FWER rule

**Phase:** RS
**Depends on:** none
**Estimated size:** medium
**Files to create/modify:**

- src/rfm_pipeline/manuscript_stages.py
- src/rfm_pipeline/__init__.py
- tests/alignment/test_RS_S01_exact_fwer.py (gate authored test-first; DO NOT weaken)

**Context.** The existing `fwer_max_stat` path selects via
`observed_scores > quantile(max_null, 1-alpha)`, which is not the exact
finite-B critical value and can over-reject at small B. Add the
Westfall–Young single-step maxT adjusted-p rule, which controls FWER at any
finite B under the complete null.

**Requirements:**

- Add `maxt_adjusted_pvalues(observed_scores, null_statistics) -> np.ndarray`
  computing, per pair j, `p_adj_j = (1 + #{b : max_over_pairs(null_b) >= obs_j}) / (B + 1)`
  where `max_over_pairs(null_b)` is the row maximum of `null_statistics`.
  Validate shapes; `null_statistics` is `(B, n_pairs)`, `observed_scores` is `(n_pairs,)`.
- Extend `multiplicity_controlled_interaction_selection` with
  `method="fwer_max_stat_exact"`: returns `(selected, p_values, threshold)` where
  `p_values` are the maxT-adjusted p-values, `selected = p_values <= alpha`, and
  `threshold` is the score-space critical value (the `floor(alpha*(B+1))`-th
  largest row-max of the null, or `None` if `floor(alpha*(B+1)) == 0`).
- Export both symbols from `__init__.py`.
- Do not change the existing `fwer_max_stat` or `bh_fdr` behavior.

**Acceptance criteria (`test_RS_S01_*`):**

- Exact formula on a fixed tiny example (hand-computed adjusted p-values).
- Adjusted p-values are monotone non-increasing in the observed score.
- `method="fwer_max_stat_exact"` selects exactly the pairs with `p_adj <= alpha`.
- **Global-null empirical FWER control:** over >= 500 complete-null replicates
  (obs and null drawn from one exchangeable draw), the empirical family-wise
  rejection rate is `<= alpha + 3*SE` at `alpha=0.2, B=199`. Pure-numpy; fast.
- Input-validation errors for bad shapes / alpha.

**Fixture requirements:** deterministic RNG seeds; no external data.

### Slice RS-S02: Prespecified scenario manifest + recovery estimands + comparators

**Phase:** RS
**Depends on:** RS-S01
**Estimated size:** large
**Files to create/modify:**

- src/rfm_pipeline/recovery_study.py (new)
- src/rfm_pipeline/baselines.py
- src/rfm_pipeline/__init__.py
- tests/alignment/test_RS_S02_recovery_study.py

**Context.** Turn the existing `synthetic_dgp` / `stress_tests` scaffolding into a
prespecified recovery study with fixed manifest and formal estimands, plus the
missing competitive comparators.

**Requirements:**

- `prespecified_recovery_scenarios() -> list[RecoveryScenario]` returning the
  seven fixed, seeded scenarios (frozen before any run): global-null,
  interaction-null (main/nonlinear signal, no true interactions),
  sparse-strong hierarchical, weak-signal (>=2 SNR levels), correlated/redundant
  predictors, pure-interaction (negligible marginals), nonlinear +
  > =1 misspecified transform outside the declared library. Each carries its
  > `SyntheticDGPSpec`, seed, and known `DGPTrueSupport`. Reuse `synthetic_dgp`.
- `recovery_estimands(true_support, selected_support) -> dict` reporting, per
  family (main / interaction / transformation) and whole-support:
  precision, recall/power, false-discovery proportion, exact-support-recovery
  (bool), selected-support size. Interaction-pair metrics use unordered pairs.
- `empirical_interaction_fwer(false_pair_flags) -> dict` returning the
  proportion of replicates with >= 1 false interaction pair and a binomial
  (Wilson) confidence interval; plus per-family mean false-selection counts.
  Do not label FDR/per-comparison/average counts as FWER.
- Add `OracleOLSBaseline` (fits OLS on the *planted* support; explicitly an
  unattainable diagnostic) and a nonlinear predictive surrogate baseline (e.g.
  gradient-boosted trees) to `baselines.py`, both honoring `BaselineProtocol`.
  Confirm a multitask/sparse-linear comparator is present (`ElasticNetBaseline`).
- All functions are case-study-agnostic (arbitrary X/Y, no BSM constants).

**Acceptance criteria (`test_RS_S02_*`):**

- Exactly 7 scenarios; each declares its planted support; the global-null
  scenario plants no interactions; the pure-interaction scenario plants
  interactions with ~zero main effects; the nonlinear scenario includes a
  misspecified transform flagged in the spec.
- `recovery_estimands` on a hand-built (true, selected) pair returns correct
  precision/recall/FDP/exact-recovery/size per family.
- `empirical_interaction_fwer` on a fixed flag vector returns the correct
  proportion and a valid Wilson interval covering it.
- `OracleOLSBaseline` and the nonlinear surrogate fit/predict on synthetic data
  and are returned by the comparison harness schema.

**Fixture requirements:** small synthetic multi-output datasets; fixed seeds.

### Slice RS-S03: Small local recovery run + released artifacts

**Phase:** RS
**Depends on:** RS-S02
**Estimated size:** large
**Files to create/modify:**

- scripts/run_recovery_study.py (new)
- tests/alignment/test_RS_S03_recovery_run.py

**Context.** Execute the prespecified study end-to-end at a documented reduced
scale that runs on a laptop, producing the released aggregate tables, figure
data, and a reproduction log. The empirical interaction-FWER is measured on the
actual interaction-discovery + `fwer_max_stat_exact` selection stage; the
mathematical guarantee is validated separately in RS-S01.

**Requirements:**

- `scripts/run_recovery_study.py` runs each prespecified scenario at a small,
  explicitly-documented scale (e.g. ~30 inputs, ~40 outputs, ~2,000 train rows,
  B=199, null-FWER replicates ~100, alternative replicates ~20 — chosen so the
  whole study runs in minutes and each stage's candidate-family logic is
  unchanged). It records, per scenario: what each stage retains, recovery
  estimands under alternatives, and the global-null empirical interaction-FWER
  with its binomial CI. Writes to `outputs/recovery_study/`:
  `fwer_calibration.csv`, `recovery_estimands.csv`, `stage_retention.csv`,
  `comparator_metrics.csv`, `figure_data/*.csv`, and `reproduction_log.md`
  (manifest: seeds, scales, versions, success criteria, timestamp).
- Deterministic under a fixed master seed; `--help`; `--quick` smoke mode used
  by the test at a tiny scale.
- The reduced scale is documented in `reproduction_log.md` as a deliberate
  reduction that preserves correlated inputs, binaries, multivariate responses,
  PCA reduction, structured discovery, and train-only selection.

**Acceptance criteria (`test_RS_S03_*`):**

- Running the driver in `--quick` mode writes all named artifacts.
- Global-null empirical interaction-FWER (with CI) is reported and its point
  estimate does not exceed `alpha + 3*SE`.
- Under the sparse-strong scenario, whole-support recall exceeds a modest floor
  and the oracle-OLS comparator is reported; under the global-null scenario the
  mean false interaction count is small.
- Artifacts are schema-valid CSVs; `reproduction_log.md` documents the scale
  reduction.

**Fixture requirements:** runs the driver's `--quick` mode in a tmp dir.
