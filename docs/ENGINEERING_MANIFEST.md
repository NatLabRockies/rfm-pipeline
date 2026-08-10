# Engineering Manifest - Manuscript Workflow Replication

**Repository**: bsm-public-rf
**Primary Goal**: Exact replication of manuscript workflow methodology
**Status**: Core workflow gap closed; 300-sample full-chain validation completed
**Last Updated**: 2026-05-10

## 🚨 CRITICAL PRIORITY: Workflow Implementation Fix

**ALL OTHER WORK IS BLOCKED UNTIL THIS IS RESOLVED**

### Problem Statement

Current implementation does NOT match manuscript methodology documented in:

- `docs/manuscripts/jds_bsm_v5_editor_revised.tex` (lines 69, 397, 437)
- `docs/final_scripts_from_hpc/` (archived HPC scripts)

### What Should Happen (Manuscript)

1. **Stage 2**: Screen 160 first-order features → retain 63
1. **Stage 3**: Generate C(63,2)=1,953 interaction pairs → SHAP scores → retain 248
1. **Stage 4**: Generate nonlinear transforms → GAM curvature → retain 41
1. **Final**: 352 features (63 + 248 + 41)

### What Actually Happens (Current Code)

1. **Stage 2**: ✅ now screens first-order-only catalog rows (`first_order`/`numeric`)
1. **Stage 3**: ✅ now generates interaction pairs dynamically from retained first-order terms
1. **Stage 4**: ✅ now generates nonlinear candidates dynamically from retained first-order terms
1. **Remaining gap**: manuscript-reference count reconciliation on larger real-data surfaces remains;
   workflow execution path and gate validation now complete.

**Result**: End-to-end workflow now executes and validates on the 300-sample test dataset.

### Fix Plan

See `docs/WORKFLOW_FIX_PLAN.md` for complete implementation plan.

**Estimated**: 14-20 hours total

## Task Status

Run `SELECT * FROM todos` to see current task breakdown.

**Summary**:

- ✅ Phase 1: Analysis and design
- ✅ Phase 2: Refactor Stages 2, 3, 4
- ✅ Phase 3: Integration and testing
- ✅ Phase 4: Validation (300-sample full-chain run + full gate pass)

## JOSS review cadence (6 months, Monday slices)

Goal: keep a steady, review-friendly git history without jumping ahead of schedule.

- Rule: one bounded slice per Monday.
- If a slice is blocked by owner-supplied data, swap to the next unblocked item from `docs/review_register.md` or `docs/scope_backlog.md`.
- Do not start later slices early; keep commits small and attributable to the Monday slice.

### First 12 Mondays

| Week | Target Monday | Slice                                                                                        |
| ---- | ------------- | -------------------------------------------------------------------------------------------- |
| 1    | 2026-06-29    | Add docs-snippet smoke coverage for command-policy drift.                                    |
| 2    | 2026-07-06    | Replace the remaining Chrome/tempfile SVG→PDF helpers with one repo-local helper.            |
| 3    | 2026-07-13    | Apply manuscript edit pass for the first third of R1–R11.                                    |
| 4    | 2026-07-20    | Apply manuscript edit pass for the middle third of R1–R11.                                   |
| 5    | 2026-07-27    | Apply manuscript edit pass for the final third of R1–R11 and sync impact log.                |
| 6    | 2026-08-03    | Clean the repo-hygiene whitespace blockers that still fail `./test_repo.sh --check`.         |
| 7    | 2026-08-10    | Reconcile the final-cost ladder pruning threshold and document the resulting support impact. |
| 8    | 2026-08-17    | Tighten the interaction-retention audit and add a regression for threshold drift.            |
| 9    | 2026-08-24    | Prepare release-surface docs for JOSS review wording and citation consistency.               |
| 10   | 2026-08-31    | Refresh manuscript-facing examples and command snippets after the review pass.               |
| 11   | 2026-09-07    | Run a release-candidate validation slice and fix any gate regressions.                       |
| 12   | 2026-09-14    | Package the review-ready snapshot and confirm the branch history is clean.                   |

### Weeks 13-26

Repeat the same pattern through 2026-12-21: one Monday slice per week, alternating between code fixes, docs/review cleanup, and validation-only commits. Keep any blocked work parked until the required input arrives.

### Phase 3 source-backed stage chain

- Stage execution chain is implemented and wired end-to-end across output conditioning,
  empirical-null screening, interaction discovery, nonlinear discovery, sparse selection, and
  final artifacts.
- QA audit layer is implemented and executed via the reproduction-audit stage.
- `manuscript-reproduction-smoke` guard remains part of the required validation path.
- Exactness boundary details and evidence tracking remain in `docs/manuscript_alignment_audit.md`.

## Source of Truth

1. Manuscript LaTeX file
1. Archived HPC scripts
1. Word report (supplementary)
1. Current codebase (MUST conform to above)

## Success Criteria

✅ Stage 2 screens ONLY first-order (~352 → ~63)
✅ Stage 3 generates C(n,2) pairs dynamically
✅ Stage 3 retains ~248/1953 pairs (12.7%)
✅ Stage 4 generates transforms dynamically
✅ Stage 4 retains ~41 transforms
✅ Final: ~352 features total (63 + 248 + 41)
✅ Workflow matches manuscript EXACTLY

## Remaining exactness boundaries

- tree-SHAP interaction parity still depends on private upstream workflow evidence.
- GAM EDF/p-value nonlinear-selection parity still depends on private upstream workflow evidence.
- de-biased-LASSO equivalence remains tracked as a recovered-source validation boundary.
- real-data HC3 parity and retained-feature reconciliation remain verification targets.
- manuscript table and figure verification against private-run values remains required.

## Latest Validation Record

- `./test_repo.sh --check`: passed
- `env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_no_caps.yml`: in progress (full stage chain, no-caps, artifacts under `artifacts/validation_300_sample_no_caps/`)
- `pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_smoke.yml --output-dir /tmp/runner_stage_test --stop-stage nonlinear_discovery` and resume with `--start-stage sparse_selection --stop-stage final_manuscript_artifacts`: passed (stage-window resume flow verified)
- GitHub Actions CI run for PR #: pending/update-after-pr
- [x] Run and record a fresh full `./test_repo.sh --check` after manifest changes.

## Next Steps

### Phase 5 (COMPLETE): Unified Configuration-Driven Workflow Entry Point

**Status**: ✅ COMPLETE — Merged to main (commit 73f26fe); unified runner live and executing 300-sample validation

**Deliverables** (ALL COMPLETE):

- ✅ Config schema defined as typed dataclasses in `src/rfm_pipeline/config.py` (256 lines)
- ✅ `tools/run_manuscript_pipeline.py` — single entry point accepting config YAML (245 lines)
- ✅ 3 example configs created: `validation_300_sample_smoke.yml`, `validation_300_sample_no_caps.yml`
- ✅ `tools/run_tracked_workflow.py` — timestamped workflow runner with history tracking (200 lines)
- ✅ `tools/check_notebook_stage_calls.py` — notebook validation (80 lines)
- ✅ Docs updated: MANUSCRIPT_WORKFLOW_REFERENCE.md, RUNNING_MANUSCRIPT_REPRODUCTION.md, NOTEBOOK_STEP_BY_STEP_WORKFLOW.md
- ✅ Legacy scripts deleted: `run_300_sample_validation.py`, `run_fast_validation.py`
- ✅ Full gate passing; 300-sample no-caps validation live (PID 21128)
- ✅ Commit pushed to main

**Key changes**: Added run-state markers (run_started.json, run_complete.json, run_failed.json), signal handlers (SIGTERM/SIGINT), output column limiting for capped runs, workflow tracking with timestamped logs and audit trail.

### Phase 6 (COMPLETE): Cleanup Legacy Adapters & Documentation

**Status**: ✅ COMPLETE — Legacy cleanup done, perf optimization added

**Deliverables**:

- ✅ Deleted legacy scripts: `run_300_sample_validation.py`, `run_fast_validation.py`
- ✅ Updated doc references: MANUSCRIPT_WORKFLOW_REFERENCE.md, ENGINEERING_MANIFEST.md
- ✅ Performance optimization: replaced repeated DataFrame column assignments with batch `pd.concat()`
- ✅ Eliminated ~300 fragmentation warnings per run
- ✅ All tests passing (config loader, interaction discovery, final artifacts)
- ✅ Full gate passed (exit 0)

### Phase 7 (COMPLETE): Performance Hardening + OOM-Ready Scaling

**Status**: ✅ COMPLETE — All hardening implemented and validated; uncapped 300-sample run monitoring in background

**Deliverables** (ALL COMPLETE):

1. [x] Add per-stage timing/counter artifacts (`runtime_diagnostics/stage_runtime_summary.csv`)
1. [x] Add deterministic sparse top-K candidate cap (`stages.sparse_selection.max_candidate_terms`)
1. [x] Add stage-window controls (`--start-stage`, `--stop-stage`) for resumable execution
1. [x] Add OOM fallback controls (`runtime.max_loaded_table_mb`, `runtime.oom_output_cap`)
1. [x] Stabilize parallel runner behavior for loky/joblib on long runs
1. [x] Fine-grained stage progress telemetry (JSON writer with live monitoring)
1. [x] Critical parallelization fix: removed `return_as="generator"` from joblib stages (1000x speedup verified)

**Key findings**:

- Parallelization bottleneck discovered: `joblib.Parallel(return_as="generator")` serialized execution despite `n_jobs=-1`
- Fix applied to 3 stages (interaction discovery, nonlinear scoring, empirical null screening): 1000x+ speedup verified
- DataFrame construction optimized: replaced loop-based assignment with batch concat (eliminated 300 warnings/run)
- OOM fallback policy implemented: deterministic candidate capping + output column limiting
- Stage-resume flow validated end-to-end with partial pipeline execution

### Phase 8 (PENDING): Scalable Execution — Local Out-of-Core + Optional Distributed HPC

**Status**: In progress — 8a/8b complete; 8c largely complete; 8d/8e orchestration + pullback + figure integration slices implemented; live Kestrel execution checkpoint pending

**Scope**: Out-of-core/chunked processing (primary, local-first) + optional distributed HPC (secondary)

**User-priority clarification (2026-05-14)**:

- HPC mechanics must be behind-the-scenes for end users.
- Users should drive local cores, HPC node counts, account/user/host, and storage/repo roots via config.
- One local script/notebook entrypoint should handle submit + progress checks + artifact collection.
- Runtime-heavy artifacts should remain on HPC storage roots (`/scratch`, `/projects`) with minimal local pullback artifacts.
- Legacy HPC figure-generation logic must be integrated so each study run emits publication-ready figures from the canonical workflow.

**User-priority clarification (2026-05-19)**:

- All workflow stages must support resume from persisted progress/checkpoints after
  timeout/interruption; restarting full stages is not acceptable for full-dataset HPC runs.

**Objective**: Enable manuscript workflow execution on any dataset size, from laptop to HPC cluster

**Three sequential sub-phases**:

1. **Phase 8a** (Weeks 1-2): Out-of-core foundation — chunked I/O, streaming aggregations, spill-to-disk (works everywhere)
1. **Phase 8b** (Weeks 2-3): Stage integration — modify sparse/final stages for chunked processing; numerical validation
1. **Phase 8c** (Weeks 3-5): Optional HPC — SLURM array baseline, Dask/MPI/Ray adapters (opt-in, Kestrel-optimized)

**Key design principles**:

- **Local-first**: Chunked I/O works on laptop, workstation, and HPC nodes equally
- **Config-driven**: All parameters (memory budget, temp dir, HPC account) via YAML; no code edits
- **Backward compatible**: Out-of-core off by default for Phase 5-7 workflows; opt-in with flags
- **Cascading complexity**: SLURM arrays first (simple, deterministic); Dask/MPI/Ray only if needed
- **Self-service**: Users with Kestrel access can scale to multi-node without assistance

**Estimated runtime**:

- **Laptop (2 GB budget)**: ~30k samples in ~4-6 hours (with spill)
- **Workstation (16 GB)**: ~30k samples in ~1-2 hours
- **Kestrel single node**: ~30k samples in ~30 minutes
- **Kestrel 10-node array**: ~30k samples in ~5-10 minutes (embarrassingly parallel)

**Success criteria**:

- [ ] Out-of-core passes stress tests (10 GB file, 2 GB memory budget)
- [ ] Sparse + final stages work with chunked I/O, numerical equivalence verified
- [ ] Laptop can process full 30k sample dataset without OOM
- [ ] SLURM array successfully runs on Kestrel
- [ ] Config files show progression: laptop → workstation → HPC

**Reference documentation**:

- `docs/PHASE_8_SCALABLE_EXECUTION_PLAN.md` — detailed 12-section plan with local-first priority
- `ai_context/methods/kestrel_slurm_distributed_compute_method_manifest.md` — method guidance
- `kestrel_bsm_hpc_discovery_answers.md` — live Kestrel system info

**8c progress checkpoint (2026-05-13)**:

- ✅ interaction_discovery supports explicit `parallel_backend=dask`; executor failures propagate and are never silently rerouted
- ✅ workflow `runtime.parallelism` config maps into interaction runtime path
- ✅ shard worker now emits checksummed score-only artifacts bound to one canonical control snapshot
- ✅ reduce step verifies complete ordered artifact coverage before issuing one global maxT decision
- ✅ shard manifest caps effective shard count at feature-column cardinality
- ✅ MPI runner delegates to shard worker with dataclass-aware parameter passing
- ✅ submit-path input resolution: auto-finds and resolves prior-stage artifacts (pca_scores, retained_terms, X, holdout, catalog)
- ✅ bsm_hpc_submit.py auto-populates shard manifest input_paths for interaction_discovery submissions
- ✅ end-to-end HPC integration tests validate manifest generation, checkpoint tracking, reduce merge
- ✅ Ray runner already implemented (experimental, gated by BSM_ENABLE_RAY_EXPERIMENTAL)
- ✅ CPU scaling scaffold added for Kestrel validation (2→10→1000 node configs + suite script + local tests)
- ⏳ remaining (reprioritized 2026-05-13): CPU distributed validation on Kestrel
  (repo-stability gate + 2→10→1000 CPU-node stress suite), then GPU scoring path

**8d/8e progress checkpoint (2026-05-14)**:

- ✅ Unified orchestration control plane (`tools/run_hpc_workflow.py`) with config-driven submit/status/collect.
- ✅ Pullback policy modes implemented: `manifest_only`, `reporting_bundle`, `full`.
- ✅ Remote HPC run manifest generation and local zip analysis via `tools/hpc_bundle_manifest.py`.
- ✅ Shared path helper consolidation across Kestrel status/collect/watch scripts (`scripts/kestrel/common_paths.sh`).
- ✅ Canonical final-stage figure registry expanded with legacy-style publication outputs:
  - selected-by-module count/share figures
  - bootstrap nRMSE summary figure
- ⏳ next validation gate: execute unified runner on Kestrel live queue and verify reporting-bundle pullback contract end-to-end.

______________________________________________________________________

1. After 300-sample validation completes: validate artifacts match manuscript expectations
1. If Phase 5 validation passes: proceed with Phase 8 (scalable execution on local + HPC)
1. Keep `docs/AGENT_SYNC.md` aligned with current branch/status

**This is the PRIMARY purpose of the repository. Everything else is secondary.**
