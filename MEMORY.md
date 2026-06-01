# Memory

Durable project memory. Record canonical interfaces, known hazards, validation expectations, and important decisions here.

## Operator continuity requirement (2026-05-27)

Before starting work, write a pre-flight next-action note in `docs/AGENT_SYNC.md`; after execution, update `docs/AGENT_SYNC.md` and memory docs with outcomes.

Persist live publication run telemetry (study IDs, controller/array/reduce job IDs, stage status, pending reasons, failure/resubmit state, and config mapping) so recovery does not require reanalysis.

Keep manuscript figure rendering publication-readable and color-blind friendly (high contrast + color-blind-safe palette + non-color cues).

For `publication_full_dataset_distributed_20260526_short_hp1`, use the deterministic post-completion pull/verify checklist documented in `docs/AGENT_SYNC.md`.
Use `docs/AGENT_SYNC.md` as the rolling source for live shard-level interaction retention telemetry.
Latest state: `publication_full_dataset_distributed_20260526_short_hp1` is complete (`RUN_COMPLETE` present) with all six stages complete and merged outputs present.
Controller login pid `333185` is `NOT_RUNNING` after successful completion.
Job `14012760` is the completed empirical-null screening reduce job for `publication_full_dataset_distributed_20260526_short_hp1`.

- 2026-05-29: `docs/manuscripts/jds_bsm.tex` was reconciled to the verified short_hp1 manuscript values: 30,000 runs, 5% holdout, 9,954 PCA-retained outputs, 69 screened inputs, 62 retained interactions, 41 retained nonlinear terms, 132 final predictors, and holdout macro nRMSE 0.0721. The manuscript now describes the L2/BH screening method, 40th-percentile LASSO alpha choice, HC3-plus-pruning stage, and SVG figure set; unresolved TODOs remain for DOIs, Steve Peterson affiliation, acknowledgements/disclaimer, per-scenario holdout breakdown, and worst-output investigation.
- 2026-05-29: sensitivity-study Phase 1-2 scaffolding is now present. `src/rfm_pipeline/synthetic_dgp.py` provides standalone pure/BSM-structure synthetic generators with true-support tracking and manuscript-table schemas; `src/rfm_pipeline/sensitivity_study.py` provides LHS DGP/config generation, job enumeration, DataFrame serialization, and result collection; companion configs/scripts/tests live under `configs/sensitivity_study/`, `scripts/`, and `tests/`. Validation passed with targeted pytest (12 tests) plus `py_compile`/CLI-help smoke checks.
