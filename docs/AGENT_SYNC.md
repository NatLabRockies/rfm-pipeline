# Agent Sync

repo: NatLabRockies/bsm-public-rf
branch: codex/p1-upstream-artifact-externalization
base_branch: main
autonomy_tier: 3
profile: autonomous
current_milestone: close manuscript exactness gaps tracked in the alignment audit
current_slice: externalize interaction/nonlinear upstream manuscript-only artifacts
slice_status: completed
last_validation: `./test_repo.sh --check` passed
next_slice: continue P1 exactness-gap closure with private-run verification evidence for HC3/figure-table parity

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
pixi run pytest -q tests/test_manuscript_alignment_audit.py
pixi run pytest -q tests/test_manuscript_documentation_contract.py tests/test_manuscript_alignment_audit.py
```

## Full gate

```bash
./test_repo.sh --check
```

## Latest slice update

- Added `## Externalized manuscript-only upstream artifacts` in
  `docs/manuscript_alignment_audit.md` to explicitly name interaction-discovery and
  nonlinear-discovery manuscript-only source workflow references and mark public-repo artifact
  availability as "no".
- Added a contract test in `tests/test_manuscript_alignment_audit.py` to prevent removal of this
  externalization ledger section.
- Validation run sequence:
  - `pixi run pytest -q tests/test_manuscript_alignment_audit.py` (expected fail before doc update, then pass)
  - `pixi run pytest -q tests/test_manuscript_documentation_contract.py tests/test_manuscript_alignment_audit.py` (pass)
  - `./test_repo.sh --check` (pass)
- Git actions:
  - Created branch `codex/p1-upstream-artifact-externalization`
  - Committed `ba97399` with slice-scoped files only
  - Pushed branch to `origin/codex/p1-upstream-artifact-externalization`
  - Opened PR `#32` (`Externalize interaction/nonlinear manuscript-only artifacts`)
