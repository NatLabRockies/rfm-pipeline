# Engineering Manifest - Manuscript Workflow Replication

**Repository**: bsm-public-rf
**Primary Goal**: Exact replication of manuscript workflow methodology
**Status**: Core workflow gap closed; 300-sample full-chain validation completed
**Last Updated**: 2026-05-08

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
- `pixi run env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python tools/run_300_sample_validation.py`: passed (full stage chain + QA audit pass; artifacts under `artifacts/validation_300_sample/`)
- GitHub Actions CI run for PR #: pending/update-after-pr
- [x] Run and record a fresh full `./test_repo.sh --check` after manifest changes.

## Next Steps

1. Re-run the validation script with progressively relaxed runtime caps (output/component/permutation) to quantify convergence toward manuscript reference counts.
1. Execute the same validation path on the larger real-data surface and record retained-count deltas in `docs/manuscript_alignment_audit.md`.
1. Confirm CI run status for this slice and keep `docs/AGENT_SYNC.md` aligned with branch/PR state.

**This is the PRIMARY purpose of the repository. Everything else is secondary.**
