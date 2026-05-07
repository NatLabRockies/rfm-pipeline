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
