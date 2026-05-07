# Engineering Manifest

## Purpose

Define the authoritative implementation plan for Copilot and human developers.

## Autonomy profile

- Profile: set by `config/agent_policy.yaml`
- Current milestone: stabilize and validate manuscript-reproduction workflow
- Current slice: restore gate/doc contract alignment and clear validation blockers

## Non-negotiable constraints

- Use repo-local Pixi.
- Do not add compatibility shims unless explicitly requested.
- Do not weaken tests.
- Do not invent new functionality.
- Scope increases require formal request and approval.

## Protected surfaces

- public APIs
- config schemas
- data contracts
- CI workflows
- `test_repo.sh`
- package metadata
- canonical interfaces documented in `MEMORY.md`

## Priority queue

### P0 — Safety / validation / repo integrity

- [x] Ensure `test_repo.sh` exists and reflects local/CI validation.
- [x] Ensure Pixi workflow is locked and documented.
- [x] Restore engineering-manifest contract content required by tests.

### P1 — Current required milestone

- [ ] Close remaining manuscript exactness gaps tracked in the alignment audit.

### P2 — Hardening

- [ ] Add edge-case and negative tests around protected surfaces.

### P3 — Documentation and examples

- [ ] Align README, examples, and docs with tested behavior.

## Completed work

- Phase 3 source-backed stage chain is in place and documented.
- QA audit layer is published at `docs/manuscript_alignment_audit.md`.
- `manuscript-reproduction-smoke` is part of the validation task chain.

## Latest Validation Record

- `./test_repo.sh --check`: passed
- GitHub Actions CI run for PR #: required before merge when operating on a PR branch
- [x] Run and record a fresh full `./test_repo.sh --check` after manifest changes.

## Remaining manuscript-exactness gaps

The current public workflow remains intentionally transparent about unresolved scientific-equivalence items:

- tree-SHAP interaction equivalence is not yet ported.
- GAM EDF/p-value nonlinear diagnostics are not yet ported.
- de-biased-LASSO manuscript equivalence is not yet fully validated.
- real-data HC3 interval parity remains pending.
- table and figure verification against frozen private-manuscript outputs remains pending.

## Deferred backlog

Use `docs/scope_backlog.md` for out-of-scope opportunities.
