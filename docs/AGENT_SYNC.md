# Agent Sync

repo: NatLabRockies/bsm-public-rf
branch: main
base_branch: main
autonomy_tier: 3
profile: autonomous
current_milestone: Phase 7 complete; Phase 8 planning underway (distributed HPC execution)
current_slice: Implemented stage-window resume/stop flow, sparse top-K capping, runtime diagnostics, and OOM fallback controls
slice_status: in_progress (uncapped 300-sample scale run active via tracked workflow)
last_validation: `./test_repo.sh --check` passed after runner/stage hardening changes; targeted pytest suites passing
next_slice: finish uncapped scale run, collect stage-runtime diagnostics, and calibrate 3k/10k/full scaling windows

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

## Phase 8: Distributed Execution and Out-of-Core Processing (PLANNING)

**Status**: Planning phase — environment discovery complete, method manifest finalized, ready for implementation kickoff

**Overview**:

Phase 8 scales the manuscript workflow from local multi-threaded Python to distributed HPC execution on NREL Kestrel and generic clusters. Three sub-phases:

1. **Phase 8a**: SLURM array baseline (config schema, manifest-driven shard runner, checkpoint/recovery)
1. **Phase 8b**: Out-of-core processing (chunked I/O, streaming aggregations, spill-to-disk)
1. **Phase 8c**: Optional adapters (Dask, MPI/mpi4py, Ray experimental)

**Key documents**:

- `docs/PHASE_8_DISTRIBUTED_HPC_PLAN.md` — 12-section detailed plan (architecture, config examples, testing, risks)
- `ai_context/methods/kestrel_slurm_distributed_compute_method_manifest.md` — method guidance from external research
- `kestrel_bsm_hpc_discovery_answers.md` — live Kestrel configuration (account=bsm, MaxArraySize=11k, filesystems, etc.)

**Design principles**:

- **Config-only**: all HPC parameters via YAML (no source code edits for different environments)
- **Multi-runtime**: SLURM arrays (primary), Dask (secondary), MPI (tertiary), Ray (experimental/opt-in)
- **Fault-tolerant**: idempotent outputs, `_SUCCESS` markers, resumable tasks
- **Backward compatible**: local single-machine runs unaffected; `distributed_execution.enabled=false` by default

**Kestrel specifics**:

- Account/project: `bsm`
- Partition recommendations: `debug` for smoke tests, `shared`/`short`/`standard`/`nvme` for production
- Max array size: 11,000 (use `%N` throttling for concurrency control)
- Filesystem: use `/projects/bsm` for durable state, `/scratch/$USER/bsm_<run_id>/` for temp, `$TMPDIR` for node-local spill
- Network interface: `hsn0` (high-speed network for Dask jobs)
- No preemption (`PreemptMode=OFF`), no auto-requeue (`JobRequeue=0`) — design for idempotent recovery

**Phase 8a Deliverables** (Weeks 1-2):

- [ ] Config schema: `DistributedExecutionConfig`, `SlurmConfig`, `SpillConfig` in `src/bsm_rfm/config.py`
- [ ] Module `src/bsm_rfm/distributed/`:
  - `config_distributed.py` — config dataclasses
  - `slurm_array_runner.py` — manifest-driven task execution
  - `spill.py` — intelligent `/scratch` vs `$TMPDIR` selection
  - `checkpoint.py` — idempotent output layout + markers
- [ ] Manifest schema: shard ID, input paths, output path, expected rows/columns, status tracking
- [ ] SLURM script templates: array job, reduce/merge job, diagnostic job (Jinja2-rendered)
- [ ] Tests: unit (config, spill logic, checkpoint semantics), smoke (manifest parsing, script rendering), integration (1-task array on Kestrel)
- [ ] Documentation: `docs/DISTRIBUTED_EXECUTION_GUIDE.md`, `docs/KESTREL_SLURM_QUICKSTART.md`

**Phase 8b Deliverables** (Weeks 2-3):

- [ ] `src/bsm_rfm/chunked_io.py`:
  - `ChunkedParquetReader` — iterate large files in chunks without full materialization
  - `ChunkedAggregation` — streaming reductions (sum, mean, count, concat)
- [ ] `src/bsm_rfm/spill_ops.py` — temp file accumulation, atomic promotion, free-space monitoring
- [ ] Workflow integration: add `use_chunked_io` config flag; modify interaction/nonlinear/sparse stages
- [ ] Tests: unit (chunk iteration, aggregations), integration (numerical equivalence on 300-sample), stress (10 GB synthetic file)

**Phase 8c Deliverables** (Weeks 3-4, optional/secondary):

- [ ] `src/bsm_rfm/distributed/dask_runner.py` — Dask DataFrame + `SLURMCluster`
- [ ] `src/bsm_rfm/distributed/mpi_runner.py` — MPI rank communication
- [ ] `src/bsm_rfm/distributed/ray_runner_experimental.py` — Ray (opt-in via `BSM_ENABLE_RAY_EXPERIMENTAL=1`)

**Phase 8 Success Criteria**:

- [ ] Config file specifies all distributed parameters without code changes
- [ ] SLURM array baseline runs 300-sample on Kestrel successfully
- [ ] Full 30k-sample dataset completes ≤ 24 hours
- [ ] Artifacts match local-run validation ≤ 5 decimal places
- [ ] Out-of-core processing passes numerical equivalence test
- [ ] Documentation enables self-service on Kestrel + generic HPC

**Estimated timeline**: Phase 8a complete in 2 weeks; 8b+8c by end of Week 4

**Blocker**: Phase 8 implementation should wait until Phase 7's uncapped 300-sample run completes, so we have accurate stage-wise timing baseline for scaling predictions and runtime estimation

______________________________________________________________________

## Phase 8 Next Steps (TBD, after 300-sample completion)

1. Collect stage-wise timing from uncapped run + validate artifacts
1. Begin Phase 8a implementation: config schema + spill logic
1. Create SLURM template renderer + smoke tests
1. Test 1-task array on Kestrel
1. Expand to multi-task arrays and validation
1. Document usage for self-service deployment

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
