# Agent Sync

repo: NatLabRockies/bsm-public-rf
branch: codex/p1-debiased-upstream-externalization
base_branch: main
autonomy_tier: 3
profile: autonomous
current_milestone: close manuscript exactness gaps tracked in the alignment audit
current_slice: externalize de-biased-LASSO upstream manuscript-only artifact dependency
slice_status: completed
last_validation: `./test_repo.sh --check` passed
next_slice: collect private-run verification evidence packet for HC3/table/figure parity and compare against public artifacts

## Blocked items

- None.

## Scope increase requests

- None.

## Files in scope

- `docs/manuscript_alignment_audit.md`
- `tests/test_manuscript_alignment_audit.py`
- `docs/AGENT_SYNC.md`
- `docs/ENGINEERING_MANIFEST.md`

## Targeted tests

```bash
pixi run pytest -q tests/test_manuscript_alignment_audit.py -k externalizes_interaction_and_nonlinear_source_artifacts
pixi run pytest -q tests/test_manuscript_documentation_contract.py tests/test_manuscript_alignment_audit.py
pixi run pytest -q tests/test_engineering_manifest.py tests/test_manuscript_alignment_audit.py
```

## Full gate

```bash
./test_repo.sh --check
```

## Latest slice update

- Added de-biased-LASSO stage entry under `## Externalized manuscript-only upstream artifacts` in
  `docs/manuscript_alignment_audit.md` to explicitly name recovered
  `LASSO_to_OLS_v9.ipynb` dependence and public-repo artifact unavailability.
- Expanded the externalization contract test in `tests/test_manuscript_alignment_audit.py`
  to require empirical-null and de-biased-LASSO upstream-source disclosure alongside
  interaction/nonlinear entries.
- Validation run sequence:
  - `pixi run pytest -q tests/test_manuscript_alignment_audit.py -k externalizes_interaction_and_nonlinear_source_artifacts` (expected fail before doc update, then pass)
  - `pixi run pytest -q tests/test_manuscript_documentation_contract.py tests/test_manuscript_alignment_audit.py` (pass)
  - `pixi run pytest -q tests/test_engineering_manifest.py tests/test_manuscript_alignment_audit.py` (pass)
  - `./test_repo.sh --check` (pass)
- Git actions:
  - Created branch `codex/p1-debiased-upstream-externalization` from clean `main`
