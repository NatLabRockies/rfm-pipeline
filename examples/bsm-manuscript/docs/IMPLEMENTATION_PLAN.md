# BSM execution plan

## Current state

G0, A, and B are `OPEN`. This worktree contains only the BSM-side
pre-execution repair defined by
`docs/execution_control/g0b_preexecution_control.md`. It does not authorize
calibration, scheduler, HPC, production, holdout, release-artifact, or
manuscript-result work.

The corrected manuscript artifact bundle remains a separate historical release
surface. Its reported values are not inputs, acceptance targets, or evidence
for the recovery-study controls.

## Implemented pre-execution slice

- a pinned, tracked control snapshot and manifest;
- one strict `configs/method_contract.yaml` source for phase, schema,
  screening, interaction, confidence, DGP, and terminal controls;
- a scenario-keyed development seed ledger;
- a typed 160-column DGP with four binary cells in both splits and bounded
  row-level heteroscedasticity;
- typed truth, terminal-ledger, empty-family, one-pair, null-denominator, and
  truthful-manifest checks;
- removal of unsupported generated recovery-study result artifacts.

## Required before any later execution gate

1. Complete the paired generic reducer/production-adapter repair at its pinned
   clean commit, including persisted multi-shard identity validation.
2. Obtain independent G0/A review of the common control snapshot, contracts,
   negative tests, and cross-repository integration.
3. Freeze the separate confirmatory seed ledger and all acceptance evidence.
4. Only then consider the bounded scheduler pilot required by the controlling
   snapshot. Gate B confirmation remains separate work.
