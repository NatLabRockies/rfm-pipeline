# AGENTS.md — Autonomous Development Profile

## Autonomy level

This repository is configured for autonomous Copilot development.

Copilot should manage repo state and next-step selection from:

- `docs/ENGINEERING_MANIFEST.md`
- `docs/AGENT_SYNC.md`
- tests and validation output
- live repo state

Do not ask what next when the repo contains enough information.

## Operating mode

Copilot may:

- create/manage task branches
- execute multiple related slices in a milestone
- write tests/source/docs/configs within scope
- run targeted tests and checkpoint full gates
- update `docs/AGENT_SYNC.md`
- update the engineering manifest and decision log when state changes
- record out-of-scope discoveries in `docs/scope_backlog.md`
- use `git-maintainer-autonomous` when `config/agent_policy.yaml` allows it

Copilot must:

- use repo-local Pixi
- follow the engineering manifest
- preserve protected surfaces unless the slice requires changing them
- write/update focused tests before implementation
- observe expected targeted failure before implementation
- avoid red commits
- avoid compatibility shims unless explicitly requested
- avoid test weakening
- prevent scope drift

## Scope control

New functionality requires explicit authorization unless it is already in the manifest, issue acceptance criteria, selected TODO, or failing test.

Scope increase requires a formal request with justification.

## Git control

This profile allows autonomous Git control when `config/agent_policy.yaml` enables it.

The autonomous Git maintainer may execute branch, stage, commit, push, PR, merge, and branch cleanup commands according to repo policy.

It must not commit unrelated changes, secrets, caches, or invalid state. It must record Git actions in `docs/AGENT_SYNC.md`.

## Stop gates

Stop and ask before:

- scope expansion not approved by manifest/policy
- dependency changes unless policy allows them
- public API/schema/data contract changes not already in manifest
- compatibility shims
- repeated validation failure after two repair attempts
- unresolved conflict between manifest and live repo

## Review and audit findings

PR review, bug, code-quality, and security findings must be persisted in `docs/review_register.md`. Blockers and required follow-up work must also update `docs/AGENT_SYNC.md` and, when they affect milestone priority, `docs/ENGINEERING_MANIFEST.md`. Non-blocking out-of-scope findings belong in `docs/scope_backlog.md`.
