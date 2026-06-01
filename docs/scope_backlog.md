# Scope Backlog

Record useful but out-of-scope discoveries. Do not implement these unless promoted into the engineering manifest or explicitly requested.

## P0 POST-VALIDATION: Remove legacy dataset-specific scripts and adapters

- Source / why noticed: Phase 5 refactor goal is to replace all hardcoded dataset scripts (run_300_sample_validation.py, etc.) with config-driven unified entry point. Now that integration is complete and unified entry point is running, legacy code is redundant.
- Reason / justification: This is a new repo with no external users; we do not need to maintain old poor designs or backward compatibility. Removing legacy adapters and scripts will clean up codebase, reduce confusion, and remove multiple entry points. Clean design is: one command (`tools/run_manuscript_pipeline.py`), many configs.
- Blocks current slice: No (validation running with new entry point; old scripts not in use)
- Priority estimate: High — must be done before merge to production/release
- Affected files/modules: `tools/run_300_sample_validation.py`, `src/rfm_pipeline/manuscript_runtime.py` (legacy config loader), legacy config loading patterns in `manuscript_stages.py` if any
- Risk if ignored: Codebase confusion; multiple entry points; users may accidentally run old scripts; outdated documentation
- Tests required if promoted: Verify all tests pass with legacy code removed; ensure unified entry point covers all cases that old scripts covered
- Recommendation: Promote to next milestone (Phase 6) immediately after 300-sample validation completes and artifacts are validated. Should be straightforward cleanup: delete old scripts, verify tests still pass, update docs to point only to unified entry point.

______________________________________________________________________

## Candidate: Parallelize and cache final-stage ablation table computation

- Source / why noticed: Observed during 300-sample no-caps validation run (2026-05-09). Final stage ran for 5+ hours due to ablation table construction.
- Reason / justification: `_compute_ablation_table` calls `build_manuscript_feature_design` 3× from scratch for up to ~1300 complex features (interaction and nonlinear), fits OLS 3× across 9466 outputs, and runs 4 serial `bootstrap_macro_nrmse_ci` loops of 200 iterations each. Design matrix should be built once and subsetted; bootstraps should be parallelized with joblib.
- Blocks current slice: No (pipeline still completes; just slowly)
- Priority estimate: High — must be resolved before any full-dataset run is feasible
- Affected files/modules: `src/rfm_pipeline/manuscript_stages.py` (`_compute_ablation_table`, `_fit_ablation_ols_nrmse`), `src/rfm_pipeline/metrics.py` (`bootstrap_macro_nrmse_ci`)
- Risk if ignored: Full-dataset runs will be prohibitively slow in the final stage; 5+ hours just for ablation even on 300 samples
- Tests required if promoted: Regression test that ablation table produces same results with cached design vs rebuild; test that parallelized bootstrap matches serial within tolerance
- Recommendation: Promote into Phase 5 refactor milestone alongside config-driven runner. Fix in same slice since both touch `manuscript_stages.py`.

______________________________________________________________________

## Template

### Candidate: title

- Source / why noticed:
- Reason / justification:
- Blocks current slice: yes/no
- Priority estimate:
- Affected files/modules:
- Risk if ignored:
- Tests required if promoted:
- Recommendation:
