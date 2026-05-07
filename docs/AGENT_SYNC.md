# Agent Sync

repo: NatLabRockies/bsm-public-rf
branch: codex/p1-upstream-artifact-externalization
base_branch: main
autonomy_tier: 3
profile: autonomous
current_milestone: close manuscript exactness gaps tracked in the alignment audit
current_slice: externalize private-run verification evidence requirements
slice_status: completed
last_validation: `./test_repo.sh --check` passed
next_slice: validate empirical-null screening equivalence against recovered Delta-null workflow or externalize retained-term artifact source of truth

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
pixi run pytest -q tests/test_manuscript_alignment_audit.py -k private_run_verification_evidence_requirements
pixi run pytest -q tests/test_manuscript_documentation_contract.py tests/test_manuscript_alignment_audit.py
pixi run pytest -q tests/test_engineering_manifest.py tests/test_manuscript_alignment_audit.py
```

## Full gate

```bash
./test_repo.sh --check
```

## Latest slice update

- Added `## Private-run verification evidence ledger` in
  `docs/manuscript_alignment_audit.md` to explicitly enumerate pending private-run evidence for:
  HC3 retained/dropped-feature parity, final coefficient parity, and manuscript Table/Figure parity.
- Added a contract test in `tests/test_manuscript_alignment_audit.py` to prevent removal/regression
  of the private-run evidence ledger section.
- Validation run sequence:
  - `pixi run pytest -q tests/test_manuscript_alignment_audit.py -k private_run_verification_evidence_requirements` (expected fail before doc update, then pass)
  - `pixi run pytest -q tests/test_manuscript_documentation_contract.py tests/test_manuscript_alignment_audit.py` (pass)
  - `pixi run pytest -q tests/test_engineering_manifest.py tests/test_manuscript_alignment_audit.py` (pass)
  - `./test_repo.sh --check` (pass)
- Git actions:
  - Continue on branch `codex/p1-upstream-artifact-externalization`
  - Existing open PR: `#32` (`Externalize interaction/nonlinear manuscript-only artifacts`)
