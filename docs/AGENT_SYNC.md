# Agent Sync

repo: NatLabRockies/bsm-public-rf
branch: codex/p1-table-figure-mapping
base_branch: main
autonomy_tier: 3
profile: autonomous
current_milestone: close manuscript exactness gaps tracked in the alignment audit
current_slice: map manuscript tables/figures to concrete release artifacts
slice_status: completed
last_validation: `./test_repo.sh --check` passed
next_slice: continue P1 exactness-gap closure (interaction/nonlinear externalization or verification)

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
```

## Full gate

```bash
./test_repo.sh --check
```

## Latest slice update

- Added a one-to-one mapping section in `docs/manuscript_alignment_audit.md` from manuscript
  Table/Figure labels to emitted `final_manuscript_artifacts/` table/figure source-data files.
- Added a contract test to keep that map present and explicit in future edits.
