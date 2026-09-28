# Generation-11 HPC submission-readiness plan

This plan implements the 2026-08-10 HPC audit.  It ends at
`HPC_SUBMISSION_READY` and never submits a scheduler job.

## Rules

- The authoritative scientific requirements are the current round-eight section
  of `docs/ANALYSIS_HANDOFF.md` and the 2026-08-10 HPC audit.
- Every implementation slice ends `READY_FOR_INDEPENDENT_REVIEW`; only a fresh,
  read-only reviewer may set `PASS`.
- A slice may not use historical P9/G10 state, outputs, caches, or seeds.
- Every scheduler command is generated and validated only.  `sbatch`, `srun`,
  `qsub`, and equivalent submission commands are forbidden.
- No downstream slice may begin while any listed dependency is not `PASS`.

### Slice G11-CTRL-S1: Materialize Generation-11 controls

**Depends on:** none
**Phase:** G11
**Task type:** implementation
**Repositories:** bsm-public-rf-manuscript, bsm-public-rf

Create and validate the generation-11 plan, prompt, validator, and tracked
control-bundle manifest.  Bind the bundle to the current handoff hash and the
two companion repository source identities.  Keep every scientific gate OPEN.
This slice does not implement scientific workers or dispatch HPC work.

### Slice G11-G0-S1: Freeze one campaign contract

**Depends on:** G11-CTRL-S2
**Phase:** G11
**Task type:** implementation
**Repositories:** rfm-pipeline, bsm-public-rf

Implement one hashed contract that generates the 6,400-record recovery matrix,
resolution study, fixed-family supplement, production stage contract, seeds,
retry limits, and artifact schemas.  It must reject duplicate IDs/seeds,
wrong dimensions, wrong binary types, stale hashes, and incomplete inventories.

### Slice G11-CTRL-S2: Review Generation-11 controls

**Depends on:** G11-CTRL-S1
**Phase:** G11
**Task type:** independent-review
**Repositories:** bsm-public-rf-manuscript, bsm-public-rf

Read controls and manifests without trusting prior reports.  Reject missing,
ignored-only, stale, or unbound control sources.

### Slice G11-G0-S2: Review campaign contract

**Depends on:** G11-G0-S1
**Phase:** G11
**Task type:** independent-review
**Repositories:** rfm-pipeline, bsm-public-rf

Recompute contract identity, campaign inventory, schedule nesting, and seed
uniqueness.  Mutate contract fields and verify fail-closed behavior.

### Slice G11-A-S1: Implement stage-native interaction artifacts

**Depends on:** G11-G0-S2
**Phase:** G11
**Task type:** implementation
**Repositories:** rfm-pipeline

Replace pair-range repeated full-fit work with complete-family draw-block
artifacts, persisted reducer validation, and serial/block equivalence tests.

### Slice G11-A-S2: Review interaction artifacts

**Depends on:** G11-A-S1
**Phase:** G11
**Task type:** independent-review
**Repositories:** rfm-pipeline

### Slice G11-P-S1: Implement train-freeze-predict and bootstrap paths

**Depends on:** G11-A-S2
**Phase:** G11
**Task type:** implementation
**Repositories:** rfm-pipeline, bsm-public-rf

Implement enforced training-only stages, immutable model freeze, holdout
authorization, exact eligibility ledger, and frozen-prediction bootstrap.

### Slice G11-P-S2: Review production safeguards

**Depends on:** G11-P-S1
**Phase:** G11
**Task type:** independent-review
**Repositories:** rfm-pipeline, bsm-public-rf

### Slice G11-HPC-S1: Generate the no-submit HPC package

**Depends on:** G11-P-S2
**Phase:** G11
**Task type:** implementation
**Repositories:** rfm-pipeline, bsm-public-rf

Generate content-addressed pilot, campaign, reducer, retry, telemetry,
resource-envelope, and post-run ingestion packages.  Validate commands,
dependencies, paths, storage/inode/AU arithmetic, and placeholders locally.

### Slice G11-HPC-S2: Independently review submission package

**Depends on:** G11-HPC-S1
**Phase:** G11
**Task type:** independent-review
**Repositories:** rfm-pipeline, bsm-public-rf, bsm-public-rf-manuscript

Read-only adversarial review of all `HPC_SUBMISSION_READY` criteria.  Scheduler
submission remains prohibited regardless of the review decision.
