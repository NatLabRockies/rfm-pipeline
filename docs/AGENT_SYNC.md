# Agent Sync

repo: NatLabRockies/bsm-public-rf
branch: copilot/manuscript-phase3-integration
base_branch: main
autonomy_tier: 3
profile: autonomous
current_milestone: validate end-to-end manuscript workflow on 300-sample test dataset
current_slice: complete validation runtime path and full-gate checkpoint
slice_status: complete
last_validation: 300-sample full stage-chain validation script completes in ~56s with QA audit pass and full gate `./test_repo.sh --check` passes
next_slice: run manuscript workflow on larger real-data surface (higher output/component caps) and reconcile retained-count deltas vs manuscript references

## Blocked items

- None. Full gate passes.

## Scope increase requests

- Added `shap >= 0.44` to `[dependencies]` in `pixi.toml` per explicit user instruction to match manuscript workflow exactly (overrides `allow_dependency_changes: false`).

## Files in scope

- `src/bsm_rfm/manuscript_stages.py`
- `tests/test_manuscript_interaction_discovery.py`
- `tests/test_manuscript_final_artifacts.py`
- `tools/run_300_sample_validation.py`
- `tests/test_manuscript_runtime.py`
- `docs/AGENT_SYNC.md`
- `docs/ENGINEERING_MANIFEST.md`

## Targeted tests

```bash
pixi run pytest -q tests/test_manuscript_interaction_discovery.py -k 'spec'
pixi run pytest -q tests/test_manuscript_final_artifacts.py -k 'spec'
pixi run pytest -q tests/test_manuscript_interaction_discovery.py
pixi run pytest -q tests/test_manuscript_reproduction_chain.py tests/test_manuscript_reproduction_audit.py
pixi run env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python tools/run_300_sample_validation.py
```

## Full gate

```bash
./test_repo.sh --check
```

## Latest slice update

- Added optional interaction runtime overrides in case-study config parsing:
  `permutation_count_B`, `n_tree_estimators`, `max_tree_depth`, `max_shap_samples`.
- Added optional final-artifact runtime overrides in case-study config parsing:
  `bootstrap_count`, `bootstrap_alpha`, and inferential-filter `alpha`.
- Added focused parser tests for these overrides:
  `test_interaction_discovery_spec_accepts_optional_runtime_overrides` and
  `test_final_artifact_spec_accepts_optional_runtime_overrides`.
- Hardened runtime resolution test to be CI-stable by constructing a temporary fully-resolved
  artifact config via monkeypatch instead of assuming machine-local real-data overrides exist.
- Added and validated `tools/run_300_sample_validation.py` full-chain runner for the 300-sample
  dataset with deterministic validation-time caps:
  - interaction null permutations: 5
  - tree estimators: 20
  - SHAP sample cap: 80
  - empirical-null B: 20, BH q: 1.0
  - stability resamples: 8
  - final bootstrap count: 20
  - output-column cap: 300
  - holdout split normalization (`test`/`validation` → `holdout`)
- 300-sample validation run completed end-to-end:
  - elapsed: ~56s
  - QA audit summary: `qa_status=pass`, `n_artifacts=51`, `n_missing_artifacts=0`,
    `n_empty_artifacts=0`, `n_failed_metric_checks=0`
  - artifacts written under `artifacts/validation_300_sample/`
- Milestone checkpoint gate:
  - `./test_repo.sh --check` ✅ pass
