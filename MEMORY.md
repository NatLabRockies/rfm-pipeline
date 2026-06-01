# Memory

Durable project memory. Record canonical interfaces, known hazards, validation expectations, and important decisions here.

## ⚠️ DIRECTORY RENAME REQUIRED (2026-06-01)

User must rename local directories after exiting this session:

```bash
mv ~/src/bsm-public-rf ~/src/rfm-pipeline       # rename framework repo dir
mv ~/src/bsm-public-rf-new ~/src/bsm-public-rf  # rename new BSM study repo dir
```

## Three-repo architecture (established 2026-06-01)

| Repo                                   | Local dir (after rename)       | Purpose                                 |
| -------------------------------------- | ------------------------------ | --------------------------------------- |
| NatLabRockies/rfm-pipeline             | ~/src/rfm-pipeline             | Generic pipeline package `rfm_pipeline` |
| NatLabRockies/bsm-public-rf            | ~/src/bsm-public-rf            | BSM configs + committed model artifacts |
| NatLabRockies/bsm-public-rf-manuscript | ~/src/bsm-public-rf-manuscript | LaTeX + figures                         |

### Package rename: bsm_rfm → rfm_pipeline (commit ce6768b, 2026-06-01)

- All imports updated; src/bsm_rfm/ deleted; 471 tests pass, 1 xfailed
- Notebooks use `RFM_STUDY_ROOT` env var and `_find_study_root()` (checks `configs/` + `pyproject.toml`)

### Phase 6 remaining: remove BSM-specific configs from rfm-pipeline

Configs now in bsm-public-rf — delete from rfm-pipeline after confirming safe:

- `configs/manuscript_*.yml`, `configs/local/`, `configs/datasets/real_data.yml`
- `configs/hpc/kestrel_publication_*.yml`, `configs/kestrel_final_cost_*.yml`
- Keep: `configs/manuscript_case_study_fast_sparse.yml` (dev/validation config)

## Operator continuity requirement (2026-05-27)

Before starting work, write a pre-flight next-action note in `docs/AGENT_SYNC.md`; after execution, update `docs/AGENT_SYNC.md` and memory docs with outcomes.

Persist live publication run telemetry (study IDs, controller/array/reduce job IDs, stage status, pending reasons, failure/resubmit state, and config mapping) so recovery does not require reanalysis.

Keep manuscript figure rendering publication-readable and color-blind friendly (high contrast + color-blind-safe palette + non-color cues).

## Publication run: COMPLETE (2026-05-29)

- Study: `publication_full_dataset_distributed_20260526_short_hp1` → `RUN_COMPLETE`
- All 6 stages complete; results in `artifacts/publication_full_dataset_distributed_results/`
- Key numbers: 30k runs, 69 screened inputs, 62 interactions, 132 final predictors, nRMSE 0.0721
- ⚠️ Final OLS all-outputs rerun job **14043519** submitted 2026-05-30 — verify completion before using new artifacts

## Sensitivity study HPC (last known: 2026-05-30)

- Job **14039970** = Batch 0; batch watcher PID **3198706** auto-submitting batches 1–5
- ETA all batches: ~June 4, 2026
- Wave 1 results in §7 (commit `7a74d29`); meta-regression degree-3 fitted (commit `61f9b0f`)
- ~46% "too few retained" = legitimate data points for sparse DGPs

## Manuscript state (2026-05-29)

- `docs/manuscripts/jds_bsm.tex` reconciled to verified values above
- Unresolved TODOs: DOIs, Steve Peterson affiliation, acknowledgements/disclaimer, per-scenario holdout breakdown, §7 sensitivity final numbers

## TransformDef API (commit b8c6520)

- `from rfm_pipeline import TransformDef, QUADRATIC, LOGARITHMIC, INVERSE, SQRT, EXPONENTIAL, DEFAULT_TRANSFORM_LIBRARY`
- Column naming: `{base}_{label}` (e.g. `x1_sq`, `income_log1p`)
- Legacy column names still resolve via fallback in `_materialize_feature_column()`

## Validation gate

- Full: `pixi run pytest -q` → 471 passed, 1 xfailed (ce6768b)
- Pre-push: `./test_repo.sh --check`
