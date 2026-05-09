# Scope Backlog

Record useful but out-of-scope discoveries. Do not implement these unless promoted into the engineering manifest or explicitly requested.

## Candidate: Parallelize and cache final-stage ablation table computation

- Source / why noticed: Observed during 300-sample no-caps validation run (2026-05-09). Final stage ran for 5+ hours due to ablation table construction.
- Reason / justification: `_compute_ablation_table` calls `build_manuscript_feature_design` 3× from scratch for up to ~1300 complex features (interaction and nonlinear), fits OLS 3× across 9466 outputs, and runs 4 serial `bootstrap_macro_nrmse_ci` loops of 200 iterations each. Design matrix should be built once and subsetted; bootstraps should be parallelized with joblib.
- Blocks current slice: No (pipeline still completes; just slowly)
- Priority estimate: High — must be resolved before any full-dataset run is feasible
- Affected files/modules: `src/bsm_rfm/manuscript_stages.py` (`_compute_ablation_table`, `_fit_ablation_ols_nrmse`), `src/bsm_rfm/metrics.py` (`bootstrap_macro_nrmse_ci`)
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
