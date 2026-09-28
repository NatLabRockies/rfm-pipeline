Execute exactly one Generation-11 slice from
`docs/AUTONOMOUS_ANALYSIS_PLAN_G11.md`.

Read the current section of `docs/ANALYSIS_HANDOFF.md`,
`docs/ANALYSIS_GATE_STATUS.md`, and the named repository instructions first.
Preserve dirty work, use Pixi, test behavior before implementation, and record
commands and hashes.  Never use P9/G10 results, seeds, caches, or state.

Implementation slices leave their gate `READY_FOR_INDEPENDENT_REVIEW`.
Independent-review slices start read-only and may set `PASS` only if they made
no implementation changes.  Never run scheduler submission commands.
