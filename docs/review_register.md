# Review Register

Durable record of pull request reviews, bug checks, code-quality audits, and security audits.

Agents use this file to convert review findings into planned work instead of leaving them in chat output.

## Finding statuses

- open
- planned
- in_progress
- fixed
- accepted_risk
- deferred
- not_reproducible

## Dispositions

- blocker: must be fixed before merge or release
- required_follow_up: must become a planned manifest/sync slice
- non_blocking_backlog: useful but outside current scope
- no_action: observation only

## Findings

### REVIEW-0001 — Template placeholder

- Status: deferred
- Severity: low
- Category: process
- Disposition: no_action
- Source: toolkit template
- Evidence: replace this placeholder with real review findings
- Affected files: none
- Required action: none
- Blocks merge: no
- Destination: none
- Notes: keep IDs stable and append new findings chronologically

### REVIEW-0002 — Runtime context sample-id mismatch blocks full-chain tests

- Status: fixed
- Severity: high
- Category: correctness
- Disposition: blocker
- Source: targeted integration test runs during workflow-fix milestone
- Evidence: runtime context now falls back to deterministic demo artifacts when real local overrides produce incompatible sample-id universes; manuscript runtime/stage integration tests pass after fix
- Affected files: runtime context loading path (`configs/local/manuscript_paths.local.yml` interactions with `rfm_pipeline.manuscript_runtime` / `rfm_pipeline.manuscript_stages`)
- Required action: completed in code; keep runtime-alignment fallback regression test active
- Blocks merge: no
- Destination: resolved in current Phase 3 integration slice
- Notes: preserves strict sample-id validation inside stage math while hardening notebook-context resolution.

### REVIEW-0003 — Full gate blocked by pre-existing repo-hygiene whitespace violations

- Status: open
- Severity: medium
- Category: process
- Disposition: required_follow_up
- Source: checkpoint run `./test_repo.sh --check`
- Evidence: `repo-hygiene` reports trailing whitespace in unrelated files (`COMPLETE_SETUP_GUIDE.md`, `DOCUMENTATION_STATUS.md`, `WORKFLOW_FINDINGS.md`, several `scripts/*.py`, `configs/datasets/README.md`)
- Affected files: multiple docs/scripts outside current Stage 2/3/4 slice
- Required action: run formatting cleanup for listed files (or remove from active branch) before rerunning full gate
- Blocks merge: yes (for branches requiring clean gate)
- Destination: hygiene cleanup slice before milestone merge
- Notes: not introduced by this slice, but currently prevents checkpoint gate completion.

### REVIEW-0004 — Public release audit found blocker/high release-surface regressions

- Status: fixed
- Severity: high
- Category: release_readiness
- Disposition: blocker
- Source: 2026-06-06 public release audit remediation slice
- Evidence: fixed sensitivity-study schema key drift, README quickstart/API mismatch, distributed default `bsm` values, SLURM job-name prefixes, docstring task-name drift, dataset/env var docs, package exports, citation metadata, logger namespace, and pytest slow-mark registration; verified `pixi run python -m pytest tests/test_distributed_phase8a.py tests/test_public_api.py tests/test_parallel_executor.py tests/test_sensitivity_study.py -x -q` and `pixi run python -m pytest tests/ -x -q`
- Affected files: `configs/sensitivity_study/*.yml`, `scripts/generate_sensitivity_study.py`, `README.md`, `src/rfm_pipeline/distributed/*.py`, `src/rfm_pipeline/__init__.py`, `src/rfm_pipeline/parallel/executor.py`, `src/rfm_pipeline/sensitivity_study.py`, `configs/datasets/*`, `docs/*`, `CITATION.cff`, `pyproject.toml`
- Required action: completed in code/docs; keep the new public-API/distributed-config regression checks active
- Blocks merge: no
- Destination: resolved in the 2026-06-06 release-audit cleanup slice
- Notes: `configs/local/manuscript_paths.local.yml` was checked and confirmed untracked, so H-13 required no additional repo mutation.
