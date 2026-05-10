# Engineering Manifest - Manuscript Workflow Replication

**Repository**: bsm-public-rf
**Primary Goal**: Exact replication of manuscript workflow methodology
**Status**: Core workflow gap closed; 300-sample full-chain validation completed
**Last Updated**: 2026-05-09

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
- GitHub Actions CI run for PR #: pending/update-after-pr
- [x] Run and record a fresh full `./test_repo.sh --check` after manifest changes.

## Next Steps

### Phase 5 (COMPLETE): Unified Configuration-Driven Workflow Entry Point

**Status**: ✅ COMPLETE — Merged to main (commit 73f26fe); unified runner live and executing 300-sample validation

**Deliverables** (ALL COMPLETE):

- ✅ Config schema defined as typed dataclasses in `src/bsm_rfm/config.py` (256 lines)
- ✅ `tools/run_manuscript_pipeline.py` — single entry point accepting config YAML (245 lines)
- ✅ 3 example configs created: `validation_300_sample_smoke.yml`, `validation_300_sample_no_caps.yml`
- ✅ `tools/run_tracked_workflow.py` — timestamped workflow runner with history tracking (200 lines)
- ✅ `tools/check_notebook_stage_calls.py` — notebook validation (80 lines)
- ✅ Docs updated: MANUSCRIPT_WORKFLOW_REFERENCE.md, RUNNING_MANUSCRIPT_REPRODUCTION.md, NOTEBOOK_STEP_BY_STEP_WORKFLOW.md
- ✅ Legacy scripts deleted: `run_300_sample_validation.py`, `run_fast_validation.py`
- ✅ Full gate passing; 300-sample no-caps validation live (PID 21128)
- ✅ Commit pushed to main

**Key changes**: Added run-state markers (run_started.json, run_complete.json, run_failed.json), signal handlers (SIGTERM/SIGINT), output column limiting for capped runs, workflow tracking with timestamped logs and audit trail.

### Phase 6 (IN PROGRESS): Cleanup Legacy Adapters & Documentation

**Objective**: Remove remaining references to deleted scripts; ensure unified config-driven runner is the only entry point.

**Current work**:

1. [x] Delete legacy dataset scripts (already done in commit 73f26fe)
1. [x] Update doc references in MANUSCRIPT_WORKFLOW_REFERENCE.md
1. [x] Update doc references in ENGINEERING_MANIFEST.md
1. [ ] Run full gate to confirm no tests break
1. [ ] Commit cleanup updates

______________________________________________________________________

1. After 300-sample validation completes: validate artifacts match manuscript expectations
1. If Phase 5 validation passes: proceed with Phase 7 (dataset scaling and full-dataset runs)
1. Keep `docs/AGENT_SYNC.md` aligned with current branch/status

**This is the PRIMARY purpose of the repository. Everything else is secondary.**
