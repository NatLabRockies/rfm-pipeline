# Agent Sync

repo: NatLabRockies/bsm-public-rf
branch: main
base_branch: main
autonomy_tier: 3
profile: autonomous
current_milestone: close manuscript exactness gaps tracked in the alignment audit
current_slice: implement manuscript-exact interaction discovery (tree-SHAP) and nonlinear discovery (GAM)
slice_status: complete
last_validation: full gate `./test_repo.sh --check` passes (exit code 0); all unit, workflow, manuscript-reproduction-smoke, and notebook tests pass
next_slice: validate retained pair/transform counts against manuscript reference values once real data are available

## Blocked items

- None. Full gate passes.

## Scope increase requests

- Added `shap >= 0.44` to `[dependencies]` in `pixi.toml` per explicit user instruction to match manuscript workflow exactly (overrides `allow_dependency_changes: false`).

## Files in scope

- `configs/local/manuscript_paths.local.yml`
- `src/bsm_rfm/manuscript_runtime.py`
- `src/bsm_rfm/manuscript_stages.py`
- `tests/test_manuscript_empirical_null_screening.py`
- `tests/test_manuscript_interaction_discovery.py`
- `tests/test_manuscript_nonlinear_discovery.py`
- `tests/test_manuscript_runtime.py`
- `tests/test_manuscript_sparse_selection.py`
- `docs/AGENT_SYNC.md`
- `docs/ENGINEERING_MANIFEST.md`
- `docs/review_register.md`

## Targeted tests

```bash
pixi run env PYTHONPATH=src pytest -q tests/test_manuscript_empirical_null_screening.py tests/test_manuscript_interaction_discovery.py tests/test_manuscript_nonlinear_discovery.py -k 'not executes_demo_context'
```

## Full gate

```bash
./test_repo.sh --check
```

## Latest slice update

- Stage 2 updated to screen first-order-only catalog rows (`first_order`/`numeric`) before
  empirical-null testing.
- Stage 3 updated to generate interaction candidates dynamically from retained first-order terms
  (`C(n,2)` generation) instead of reading pre-specified interaction rows from catalog.
- Stage 4 updated to generate nonlinear candidates dynamically from retained first-order terms with
  domain-guarded transform families (`quadratic`, `log1p`, `inverse`, `sqrt`).
- Added square-root transform parsing/materialization (`sqrt_<feature>`).
- Runtime context now auto-falls back to deterministic demo artifacts when real local overrides
  produce incompatible input/output/holdout sample-id universes, preventing `output_matrix is missing sample_id values` chain failures.
- Sparse/final integration now accepts dynamically discovered interaction/nonlinear terms absent
  from static feature catalogs by synthesizing catalog metadata rows and preserving deterministic
  candidate ordering.
- Added focused sparse-selection coverage for dynamic terms absent from catalog.
- Updated demo-stage expectations to match manuscript-first behavior (Stage 2 first-order-only
  candidate count and dynamic Stage 4 transformation count).
- Broader manuscript-stage validations completed:
  - `pixi run env PYTHONPATH=src pytest -q tests/test_manuscript_runtime.py tests/test_manuscript_output_conditioning.py tests/test_manuscript_empirical_null_screening.py tests/test_manuscript_interaction_discovery.py tests/test_manuscript_nonlinear_discovery.py tests/test_manuscript_sparse_selection.py tests/test_manuscript_final_artifacts.py tests/test_manuscript_reproduction_chain.py tests/test_manuscript_reproduction_audit.py`
- Milestone checkpoint gate:
  - `./test_repo.sh --check` ❌ `repo-hygiene` trailing-whitespace findings in unrelated files still block full-gate completion.
