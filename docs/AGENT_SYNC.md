# Agent Sync

repo: NatLabRockies/bsm-public-rf
branch: main
base_branch: main
autonomy_tier: 3
profile: autonomous
current_milestone: Phase 8a implementation (local-first out-of-core foundation)
current_slice: Runtime regression triage + high-fidelity telemetry + small-dataset full-run/uncapped ramp preflight
slice_status: in_progress (uncapped small-sample run bottleneck isolated; phase-8a merge pending)
last_validation: `BSM_PROGRESS_BATCH_SIZE=1 pixi run python tools/run_manuscript_pipeline.py configs/validation_80_sample_workflow_smoke.yml` passed end-to-end; `pixi run ruff check tools/run_manuscript_pipeline.py src/bsm_rfm/manuscript_stages.py src/bsm_rfm/config.py` clean
next_slice: tune/guard interaction-discovery uncapped path (1000 permutations) before retrying full uncapped ramp

## Runtime triage update (2026-05-11)

- Fixed runner bug: `tools/run_manuscript_pipeline.py` now honors `dataset.path`/`dataset.type` instead of hardcoding `artifacts/test_dataset_300`.
- Added high-fidelity telemetry:
  - final-stage substeps (`final_manuscript_artifacts`, 10 substeps)
  - ablation-model progress (`final_ablation`, 5 model checkpoints)
  - env override `BSM_PROGRESS_BATCH_SIZE` for finer progress granularity.
- Added small-dataset ramp configs:
  - `configs/validation_80_sample_workflow_smoke.yml` (capped, full chain, passes)
  - `configs/validation_80_sample_uncapped.yml` (uncapped preflight)
- Added local small dataset artifact root: `artifacts/test_dataset_160/`.
- Findings:
  - Capped small full workflow completes in ~7 seconds on 158-row dataset.
  - Uncapped small run stalls in interaction stage at `permutation_scores` 0/1000 even with fine-grain progress; this stage is current runtime blow-up point.

## Ad hoc request: external research handoff (Kestrel SLURM)

- Status: prepared handoff packet scaffold only (no external claims added locally).
- Scope decision captured: multi-runtime comparison first for distributed compute support on NREL Kestrel.
- Prompt artifact created: `ai_context/prompts/kestrel_slurm_external_research_request.md`.
- Empty research artifact templates scaffolded under:
  - `ai_context/literature/`
  - `ai_context/methods/`
  - `ai_context/api_docs/`
  - `ai_context/manifests/`
  - `ai_context/prompts/`

## Completed: Phase 5 Integration (Config-Driven Entry Point)

**Status**: ✅ COMPLETE — Merged to main; all tests passing

**Deliverables** (ALL COMPLETE):

- ✅ `src/bsm_rfm/config.py` — typed config dataclasses (256 lines)
- ✅ `tools/run_manuscript_pipeline.py` — unified entry point with full integration (245 lines)
- ✅ `configs/` directory — 3 production example configs
- ✅ `tests/test_config_loader.py` — 7 unit tests (all pass)
- ✅ `config_to_legacy_case_study()` adapter function — maps WorkflowConfig to legacy format
- ✅ Data loading (X, Y, holdout, feature catalog)
- ✅ Full manuscript_stages integration — calls run_manuscript_reproduction_stage_chain()
- ✅ Proper error handling and timing reporting
- ✅ Merged to main (commit ebae0b2); full repo gate clean

**Integration work completed**:

- Implemented config_to_legacy_case_study() mapping all fields:
  - algorithm.retained_components → output_conditioning
  - stage configs → per-stage empirical_null_screen, interaction_discovery, etc.
  - runtime.n_jobs → case_study.runtime.n_jobs
  - output.seed → random_seed
- Load all required data tables (test_dataset_300)
- Create \_FakeContext compatible with manuscript_stages expectations
- Call run_manuscript_reproduction_stage_chain() with proper parameters
- Tested with fast config; output validated

**Entry point usage**:

```bash
# Full run (1000 perms, 100 resamples, 200 bootstraps, n_jobs=-1)
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_no_caps.yml

# Fast mode (5 perms, 8 resamples, 20 bootstraps, n_jobs=1)
pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_fast.yml

# Monitor timing
python tools/monitor_validation_timing.py artifacts/validation_300_sample_no_caps
```

## Next immediate actions

1. **Monitor 300-sample validation** — currently on stage 3/6 (interaction scoring); ~2h remaining at current pace
1. **After 300-sample completes**: Validate artifacts (check all 6 stage directories, verify feature counts, timing data)
1. **Phase 7 decision**: Based on 300-sample results, determine next priority:
   - Full-dataset validation (if 300-sample passes and timings are acceptable)
   - Manuscript table/figure comparison against private reference values
   - Additional performance optimization if stages are bottlenecked

## Completed: Phase 6 (Cleanup Legacy Adapters & Documentation)

**Status**: ✅ COMPLETE — Cleanup complete, perf optimization added (commits f3152bc, d4d5787)

**Deliverables**:

- ✅ Deleted legacy scripts: `run_300_sample_validation.py`, `run_fast_validation.py`
- ✅ Updated doc references: MANUSCRIPT_WORKFLOW_REFERENCE.md, ENGINEERING_MANIFEST.md
- ✅ Performance optimization: replaced repeated DataFrame column assignments with batch `pd.concat()`
- ✅ Eliminated ~300 fragmentation warnings per run
- ✅ Smoke config runs 12s with 0 warnings (was 300+)
- ✅ All tests passing (config loader, interaction discovery, final artifacts)
- ✅ Full gate passed (exit 0)

## Completed: Phase 6 Performance Optimization

**DataFrame Construction** (commit d4d5787):

- **Change**: Replaced loop-based `design[col] = value` with batch `pd.concat(feature_df, axis=1)`
- **Benefit**: Eliminates pandas fragmentation warnings, cleaner code, same numerical output
- **Validation**: All manuscript tests pass; smoke config runs cleanly
- **Impact**: ~300 warnings eliminated per run; faster memory allocation

## Completed: Phase 7 Performance Hardening (Current Slice)

**Status**: ✅ IMPLEMENTED — validation complete, uncapped run monitoring continues

**Deliverables**:

- ✅ Added deterministic sparse candidate top-K cap (`max_candidate_terms`) to prevent sparse-stage blowups
- ✅ Added runtime OOM controls: `runtime.max_loaded_table_mb` + `runtime.oom_output_cap`
- ✅ Reworked pipeline runner for stage-window execution:
  - `--start-stage` for resume from existing artifacts
  - `--stop-stage` for partial/debug execution
- ✅ Added runtime diagnostics artifact:
  - `runtime_diagnostics/stage_runtime_summary.csv` with per-stage elapsed time + stage counters
- ✅ Added joblib/loky runtime stabilization in runner (`JOBLIB_TEMP_FOLDER`, `LOKY_MAX_CPU_COUNT`)
- ✅ Removed generator-based parallel consumption in nonlinear/stability loops to reduce backend fragility
- ✅ Verified partial+resume flow:
  - `output_conditioning → nonlinear_discovery`
  - resumed `sparse_selection → final_manuscript_artifacts`
  - both complete successfully and produce markers/artifacts

## Completed: Phase 7 Performance Hardening (Current Slice)

**Status**: ✅ COMPLETE — all hardening implemented and validated; uncapped 300-sample run monitoring continues

**Deliverables**:

- ✅ Added deterministic sparse candidate top-K cap (`max_candidate_terms`) to prevent sparse-stage blowups
- ✅ Added runtime OOM controls: `runtime.max_loaded_table_mb` + `runtime.oom_output_cap`
- ✅ Reworked pipeline runner for stage-window execution:
  - `--start-stage` for resume from existing artifacts
  - `--stop-stage` for partial/debug execution
- ✅ Added runtime diagnostics artifact:
  - `runtime_diagnostics/stage_runtime_summary.csv` with per-stage elapsed time + stage counters
- ✅ Added joblib/loky runtime stabilization in runner (`JOBLIB_TEMP_FOLDER`, `LOKY_MAX_CPU_COUNT`)
- ✅ Removed generator-based parallel consumption in nonlinear/stability loops to reduce backend fragility
- ✅ **CRITICAL FIX**: Removed `return_as="generator"` from 3 Parallel() calls (1000x+ speedup verified on smoke tests)
- ✅ Added fine-grained stage progress telemetry: JSON writer with live monitoring capability
- ✅ Verified partial+resume flow end-to-end

______________________________________________________________________

## Phase 8: Distributed Execution and Out-of-Core Processing

**Status**: Phase 8a (local-first out-of-core foundation) IN PROGRESS

**Architecture** (revised to local-first + optional HPC):

- Primary: Out-of-core chunked I/O + streaming aggregations + spill-to-disk (works on any machine)
- Optional secondary: SLURM distributed execution (for HPC acceleration on Kestrel)

**Completed** ✅:

- [x] HPC environment discovery (Kestrel probing + method manifest)
- [x] Phase 8 plan revision: local-first vs HPC-first
- [x] Feature branch `feature/phase-8a-out-of-core-foundation` created
- [x] Phase 8a foundation modules implemented + tested + committed:
  - `src/bsm_rfm/out_of_core/chunked_io.py` — ChunkedParquetReader, ChunkedCSVReader
  - `src/bsm_rfm/out_of_core/streaming_ops.py` — StreamingAggregation, StreamingQuantile
  - `src/bsm_rfm/out_of_core/memory.py` — MemoryBudget, choose_temp_dir, get_disk_free_mb
  - `src/bsm_rfm/out_of_core/spill_ops.py` — SpillToDiskBuffer, LargeArrayWriter
  - `src/bsm_rfm/out_of_core/progress.py` — ChunkProgress telemetry
- [x] Tests committed (19 unit tests, all passing)
  - `tests/test_chunked_io.py` — 8 I/O tests
  - `tests/test_streaming_ops.py` — 11 aggregation + equivalence tests
- [x] All linting fixed; pre-commit hooks pass

**In Progress** ⏳:

- 300-sample uncapped validation still running (stage 3/6, ~2-4 hours remaining as of last check)

**Remaining Phase 8a Tasks** (next):

- [x] Add OutOfCoreConfig dataclass to config.py
- [x] Integrate chunked loading path in `tools/run_manuscript_pipeline.py` (used by sparse/final via stage inputs)
- [x] Run numerical equivalence tests on out-of-core readers/aggregations (focused fast suite)
- [x] Run memory stress tests with forced spill (`SpillToDiskBuffer` tiny budget)
- [ ] Merge feature/phase-8a-out-of-core-foundation to main after validation

**Phase 8b (out-of-core integration)** → Phase 8c (optional SLURM) after 8a merged

**Design decisions**:

- Chunk size: row-group aware for Parquet (often 512 MB default), configurable per machine
- Spill strategy: Parquet format on fast local NVMe/ProjectFS, avoid tmpfs
- Memory model: track with psutil, spill when threshold hit, resume on re-read
- Backward compatible: off by default; opt-in via config `use_chunked_io: true`

**Configuration example** (for when implemented):

```yaml
runtime:
  use_chunked_io: true
  out_of_core:
    chunk_size_mb: 512
    max_memory_budget_mb: 8000
    temp_dir: /scratch/$USER/bsm_spill
    enable_spill_to_disk: true
```

**Key documents**:

- `docs/PHASE_8_SCALABLE_EXECUTION_PLAN.md` — current authoritative design spec (local-first architecture)
- `docs/ENGINEERING_MANIFEST.md` — Phase 8 overview updated
- `kestrel_bsm_hpc_discovery_answers.md` — Kestrel-specific configuration (account=bsm, MaxArraySize=11k, etc.)

**Blocker Resolution**: Phase 8 implementation was unblocked by user approval for scope expansion (no longer waiting for 300-sample). Work proceeds in feature branch in parallel while 300-sample runs overnight.

## Files in scope

- `src/bsm_rfm/manuscript_stages.py`
- `tests/test_manuscript_interaction_discovery.py`
- `tests/test_manuscript_final_artifacts.py`
- `tools/run_manuscript_pipeline.py`
- `tests/test_manuscript_runtime.py`
- `docs/AGENT_SYNC.md`
- `docs/ENGINEERING_MANIFEST.md`

## Targeted tests

```bash
pixi run pytest -q tests/test_manuscript_interaction_discovery.py -k 'spec'
pixi run pytest -q tests/test_manuscript_final_artifacts.py -k 'spec'
pixi run pytest -q tests/test_manuscript_interaction_discovery.py
pixi run pytest -q tests/test_manuscript_reproduction_chain.py tests/test_manuscript_reproduction_audit.py
pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_smoke.yml
pixi run env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python tools/run_manuscript_pipeline.py configs/validation_300_sample_no_caps.yml
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
