# Agent Sync

repo: NatLabRockies/bsm-public-rf
branch: main
base_branch: main
autonomy_tier: 3
profile: autonomous
current_milestone: stabilize repository gate and docs contract
current_slice: restore full gate pass and align manifest/sync docs
slice_status: completed
last_validation: `./test_repo.sh --check` passed
next_slice: select next manifest priority after clean baseline

## Blocked items

- None.

## Scope increase requests

- None.

## Files in scope

- `test_repo.sh`
- `docs/ENGINEERING_MANIFEST.md`
- `docs/index.md`
- `docs/AGENT_SYNC.md`

## Targeted tests

```bash
pixi run pytest -q tests/test_gate_contract.py
pixi run pytest -q tests/test_engineering_manifest.py
```

## Full gate

```bash
./test_repo.sh --check
```

## Latest slice update

- Repaired `test_repo.sh` mode/task contract to match gate tests and CI usage.
- Restored engineering-manifest required content and validation record sections.
- Added missing docs pages to Sphinx toctree to satisfy warnings-as-errors docs builds.
