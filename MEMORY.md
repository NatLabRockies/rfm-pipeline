# Memory

Durable project memory. Record canonical interfaces, known hazards, validation expectations, and important decisions here.

## ⚠️ DIRECTORY RENAME REQUIRED (2026-06-01)

**DONE** — user completed rename before this session.

- `~/src/rfm-pipeline` → rfm-pipeline repo (NatLabRockies/rfm-pipeline)
- `~/src/bsm-public-rf` → new BSM study repo (NatLabRockies/bsm-public-rf)

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

## Phase 6 removal — COMPLETE (commit 372582f, 2026-06-01)

All BSM-specific configs removed from rfm-pipeline. Generic defaults work without a case study config (`.get()` fallbacks). 461 tests pass, 1 xfailed.

## Final OLS all-outputs fix — COMPLETE (commit c5c6aa9 in bsm-public-rf)

Job 14043519 completed 2026-05-30 (7m57s). `coefficient_matrix_standardized.csv` = 23,495 rows (all outputs). Artifacts synced to `bsm-public-rf`. `hc3_wald_intervals.csv` (450MB) gitignored with regen note.

## Sensitivity study HPC (2026-06-01)

- Wave 1: job 14039970 — ~97% done (2,671/2,750 artifacts), results.csv collected
- Wave 2: job 14045231 — ~45% done (1,229/2,750 artifacts), running on shared partition
- Wave 2b (main study): job 14062332 — running, writing to `/scratch/dhetting/bsm/sensitivity_study/`
- Wave 3: job **14069433** — queued on shared partition (8h, 2,750 tasks, seeds 3000)
- Wave 3 spec: `configs/sensitivity_study/study_spec_wave3.yml`
- HPC clone at `/home/dhetting/src/bsm-public-rf` remote now points to `NatLabRockies/rfm-pipeline`

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
