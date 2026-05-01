# Engineering Manifest

This manifest is the authoritative implementation-priority ledger for this repository. It reflects
the live repository state and should be updated when work changes task status or implementation
priority. Older handoff notes in `docs/MEMORY.md` are advisory only; prefer this file for future
task selection.

## Current Repository State

- The canonical source package is `src/bsm_rfm/`.
- The canonical test suite is `tests/`.
- The manuscript notebook entrypoints are `notebooks/manuscript/00_...` through
  `notebooks/manuscript/08_...`.
- The canonical local gate is `./test_repo.sh --check`.
- CI runs `./test_repo.sh --ci` through Pixi in `.github/workflows/ci.yml`.
- Public documentation is built from `docs/index.md` with Sphinx and MyST.
- `docs/manuscript_alignment_audit.md` is the scientific-status ledger for manuscript-exactness
  claims and remaining approximation gaps.

## Non-Negotiable Constraints

- Treat the live repository as the source of truth.
- Work only on a task branch before editing.
- Preserve user work and ignored generated artifacts.
- Use test-first development for implementation work.
- Do not add compatibility shims unless explicitly requested.
- Do not claim full manuscript-exact reproduction until the alignment audit gaps are closed.
- Keep local validation and CI behavior aligned through the repository gate.

## Validation Gate

Authoritative local validation:

```bash
./test_repo.sh --check
```

Targeted validation should be run first for the changed area. Relevant Pixi tasks include:

- `pixi run lint`
- `pixi run test`
- `pixi run notebook-check`
- `pixi run manuscript-reproduction-smoke`
- `pixi run notebook-tests`
- `pixi run docs`
- `pixi run package-build`

Use `./test_repo.sh --fix` only when formatting, Markdown normalization, notebook hygiene fixes, and
transient cleanup are intended.

## Latest Validation Record

- `./test_repo.sh --check`: passed on `codex/sparse-selection-guardrail` after adding guardrail
  test and manifest updates.
- GitHub Actions CI run for PR #5: passed before merge (commit `64a97e1`).

## Completed Work

- Phase 0 manuscript contract freeze exists in `docs/manuscript_contract.md`,
  `configs/manuscript_case_study.yml`, and related tests.
- Phase 1 manuscript data/runtime contracts exist in `docs/manuscript_data_contract.md`,
  `configs/manuscript_data_contract.yml`, `configs/manuscript_paths.template.yml`,
  `configs/manuscript_runtime.yml`, and related runtime tests.
- Phase 2 manuscript notebook entrypoints and runtime/path-resolution helpers exist under
  `notebooks/manuscript/` and `bsm_rfm.manuscript_runtime`.
- Phase 3 source-backed stage chain exists in `bsm_rfm.manuscript_stages`, covering output
  conditioning, empirical-null screening, interaction discovery, nonlinear discovery,
  sparse-selection/stability filtering, final manuscript artifact generation, and the complete
  reproduction stage chain.
- The QA audit layer exists through `audit_manuscript_reproduction_outputs(...)`,
  `write_manuscript_reproduction_audit(...)`, and
  `run_manuscript_reproduction_audit_stage(...)`.
- The gate-level `manuscript-reproduction-smoke` task runs the audited deterministic demo
  reproduction path and is included in `test_repo.sh`.
- Public docs distinguish the source-backed executable scaffold from full manuscript-exact
  reproduction.

## Priority Order

### P0 - Safety, Validation, And Planning Integrity

- [x] Maintain a canonical repository gate in `test_repo.sh`.
- [x] Align CI around the local repository gate.
- [x] Add this engineering manifest as the authoritative planning document.
- [x] Run and record a fresh full `./test_repo.sh --check` after manifest changes.
- [x] Keep planning docs synchronized when manuscript-stage status changes.
- [x] Add explicit CI/local parity guardrail test for demo sparse-selection
  `final_stable_support` non-emptiness with diagnostic failure messages
  (`test_demo_sparse_selection_final_stable_support_nonempty_ci_parity_guard`).

### P1 - Release-Hardening And Public Contract Stability

- [ ] Keep public docs aligned with the implemented Phase 3 source-backed entrypoints, QA audit
  layer, and smoke gate.
- [ ] Preserve tests that prevent stale skeleton wording or exactness overclaims.
- [ ] Keep `docs/manuscript_alignment_audit.md` exposed in docs and current with implementation
  status.
- [ ] Verify example-script contracts remain aligned with `tests/test_reproducibility_example.py`.

### P2 - Manuscript-Exactness Reconciliation

- [ ] Replace, port, or externalize the tree-SHAP interaction workflow; the current public
  interaction stage is a residualized-product surrogate, not a tree-SHAP interaction
  implementation.
- [ ] Replace, port, or externalize the GAM EDF/p-value nonlinear workflow; the current public
  nonlinear stage is a residualized parametric-transform surrogate.
- [ ] Port or validate de-biased-LASSO sparse selection equivalence; the current public
  sparse-selection stage remains an EBIC/L1 surrogate with stability diagnostics.
- [ ] Perform real-data HC3 verification for final retained-feature counts, dropped-feature counts,
  and final coefficients.
- [ ] Complete one-to-one manuscript table and figure verification against manuscript labels,
  source artifacts, and expected values.

### P3 - Scientific Evidence and Workflow Justification

These items strengthen the evidentiary basis for the staged workflow and are required before the
manuscript can claim the workflow is well-supported rather than merely reported to work.

- [ ] **Ablation table**: add a compact multi-model comparison to quantify each stage's contribution.
  Required baselines:

  - mean-only baseline (intercept only);
  - main-effects-only OLS (all first-order inputs, no screening);
  - screened-only OLS (empirical-null survivors, no sparse selection);
  - penalized model (EBIC/L1 selected, before HC3 filter);
  - final OLS refit (current manuscript endpoint);
  - optional: no-interaction variant (sparse selection on first-order terms only);
  - optional: no-nonlinearity variant (sparse selection excluding transformation terms).
    Output: holdout nRMSE per model, a ranked comparison table, and a test asserting the final OLS
    refit is not worse than the mean-only baseline on the demo fixture.

- [ ] **Output-wise performance summaries**: a single aggregate nRMSE can hide bad performance on
  small-magnitude, volatile, or policy-relevant outputs. Required additions:

  - quantiles (e.g. p10, p25, p50, p75, p90) of per-output holdout nRMSE;
  - worst-output diagnostics (top-K outputs by nRMSE, flagged by name/module);
  - performance stratified by output family or BSM module where module labels are available.
    Output: a `per_output_nrmse_summary.csv` artifact and SVG diagnostic, gated by a test asserting
    median per-output nRMSE is finite and positive on the demo fixture.

### P4 - Optional Hardening

- [ ] Decide whether CI should continue rebuilding docs after `./test_repo.sh --ci`, since the gate
  already includes docs.
- [ ] Consider whether generated example artifacts should remain ignored-only outputs or receive a
  documented regeneration workflow.
- [ ] Revisit whether notebook-specific feature-expansion defaults should become a canonical
  source-driven default specification.

## Known Risks

- The deterministic demo chain is gate-tested, but it is not proof of private real-data manuscript
  exactness.
- Interaction, nonlinear, and sparse-selection stages intentionally carry conservative provenance
  fields marking private-workflow equivalence as not yet validated.
- Local private artifact paths are intentionally untracked under `configs/local/`.
- Ignored `artifacts/` and `*.zip` outputs may be present locally and should not be treated as
  tracked source state.

## Deferred Work

- Real-data dry runs require local private artifact paths in `configs/local/manuscript_paths.local.yml`.
- Scientific exactness work should proceed in small slices with tests and provenance updates for each
  stage.
- Do not reopen completed Phase 3 infrastructure as future work; future Phase 3 work should focus on
  exactness reconciliation, real-data verification, and public-claim hardening.
