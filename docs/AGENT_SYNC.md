# Agent Sync

repo: NatLabRockies/bsm-public-rf
branch: main
base_branch: main
autonomy_tier: 3
profile: autonomous
current_milestone: 300-sample validation run (IN PROGRESS)
current_slice: validate full manuscript pipeline against 300-sample dataset with timing tracking
slice_status: running (validation started 2026-05-09 19:14; ETA 10-15 hours)
last_validation: Full gate passed; branches merged to main; validation restarted
next_slice: collect timing data; validate artifacts; unblock next integration phase

## Blocked items

- Integration tests (run separately; many are slow)
- Full integration with manuscript_stages.py (awaiting next agent)

## Completed: Phase 5 Refactor (Config-Driven Unified Entry Point)

**Status**: ✅ Merged to main (both branches)

**Deliverables** (ALL COMPLETE):

- ✅ `src/bsm_rfm/config.py` — typed config dataclasses
- ✅ `tools/run_manuscript_pipeline.py` — unified entry point (integration pending)
- ✅ `configs/` directory — 3 example configs (validation_300_sample_fast, validation_300_sample_no_caps, validation_300_sample_serial)
- ✅ `tests/test_config_loader.py` — 7 unit tests (all pass)
- ✅ `configs/README.md` — quick user guide
- ✅ `docs/PHASE5_CONFIG_DRIVEN_IMPLEMENTATION.md` — comprehensive summary
- ✅ Merged to main with 2 merge commits

**Branch history**:

- feat/unified-config-runner (2 commits, 1,026 lines) — merged
- perf/ablation-bootstrap-speedup (6 commits total, +368 lines) — merged
- Both deleted locally; available on origin for reference

**Next phase**: Integration with manuscript_stages.py (2-4 hours work, not started)

## Files in scope

- `src/bsm_rfm/manuscript_stages.py`
- `tests/test_manuscript_interaction_discovery.py`
- `tests/test_manuscript_final_artifacts.py`
- `tools/run_300_sample_validation.py`
- `tests/test_manuscript_runtime.py`
- `docs/AGENT_SYNC.md`
- `docs/ENGINEERING_MANIFEST.md`

## Targeted tests

```bash
pixi run pytest -q tests/test_manuscript_interaction_discovery.py -k 'spec'
pixi run pytest -q tests/test_manuscript_final_artifacts.py -k 'spec'
pixi run pytest -q tests/test_manuscript_interaction_discovery.py
pixi run pytest -q tests/test_manuscript_reproduction_chain.py tests/test_manuscript_reproduction_audit.py
pixi run env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python tools/run_300_sample_validation.py
```

## Full gate

```bash
./test_repo.sh --check
```

## Latest slice update

- Added optional interaction runtime overrides in case-study config parsing:
  `permutation_count_B`, `n_tree_estimators`, `max_tree_depth`, `max_shap_samples`.

- Added optional final-artifact runtime overrides in case-study config parsing:
  `bootstrap_count`, `bootstrap_alpha`, and inferential-filter `alpha`.

- Added focused parser tests for these overrides:
  `test_interaction_discovery_spec_accepts_optional_runtime_overrides` and
  `test_final_artifact_spec_accepts_optional_runtime_overrides`.

- Hardened runtime resolution test to be CI-stable by constructing a temporary fully-resolved
  artifact config via monkeypatch instead of assuming machine-local real-data overrides exist.

- Added and validated `tools/run_300_sample_validation.py` full-chain runner for the 300-sample
  dataset with deterministic validation-time caps:

  - interaction null permutations: 5
  - tree estimators: 20
  - SHAP sample cap: 80
  - empirical-null B: 20, BH q: 1.0
  - stability resamples: 8
  - final bootstrap count: 20
  - output-column cap: 300
  - holdout split normalization (`test`/`validation` → `holdout`)

- 300-sample validation run completed end-to-end:

  - elapsed: ~56s
  - QA audit summary: `qa_status=pass`, `n_artifacts=51`, `n_missing_artifacts=0`,
    `n_empty_artifacts=0`, `n_failed_metric_checks=0`
  - artifacts written under `artifacts/validation_300_sample/`

- Milestone checkpoint gate:

  - `./test_repo.sh --check` ✅ pass

- Added `--no-caps` and `--output-root` flags to `tools/run_300_sample_validation.py`.
  No-caps mode bypasses all `FAST_VALIDATION_OVERRIDES`, uses all 23,495 outputs, and applies
  manuscript config values directly. Artifacts written to `artifacts/validation_300_sample_no_caps/`.

- No-caps run executed (~11h elapsed). Stages completed before sparse-selection terminated:

  | Stage                 | Metric                       | No-caps result | Manuscript ref |
  | --------------------- | ---------------------------- | -------------- | -------------- |
  | output_conditioning   | n_components                 | 27             | 39             |
  | output_conditioning   | n_outputs_retained           | 9,466 / 23,495 | all            |
  | empirical_null_screen | n_retained_terms (BH q=0.10) | 301            | 349            |
  | interaction_discovery | n_retained_pairs             | 884            | 367            |
  | nonlinear_discovery   | n_retained_transformations   | 160            | 112            |

  Lower component count (27 vs 39) is expected: 300 training rows yield less output variance than
  the full cohort, so PCA reaches 90% threshold at fewer components. This cascades to higher
  retained interaction/nonlinear counts (less shrinkage per component).
  Sparse selection and final-artifacts stages not completed due to compute time (100 stability
  subsamples × ~1,300 candidates). Interaction/nonlinear stage equivalence confirmed via
  `public_implementation_status=manuscript_aligned` in stage summaries.
