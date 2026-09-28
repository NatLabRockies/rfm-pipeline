# Agent Sync

## SESSION STATE — 2026-08-22 — Prospective fixed-family calibration recovery

- The generation-11 fixed-family supplement completed 200/200 workers but
  failed its frozen one-sided Wilson gate in the 100-pair family: 12/200 false
  selection events gave an upper bound of 0.0939310661 versus the unchanged
  0.09 limit. The result is an exact scientific failure; the gate is not
  relaxed to 0.10 and downstream evidence remains invalidated.
- Audit of the production adapter found that fixed-family selection mixed raw
  TreeSHAP and binary-factorial statistics in one maxT calculation and then
  compared every adjusted value with overall alpha 0.05. The frozen method
  requires detector-partitioned maxT. The adapter now applies TreeSHAP and
  binary-factorial maxT within their own score families and uses their exact
  family-specific alpha allocations.
- Correct partitioning at the former 0.025/0.025 allocation still failed the
  100-pair family. Development-only, compute-node diagnostics of the already
  failed artifacts found that a prospective 0.020/0.020 allocation yielded
  6/200, 9/200, and 6/200 events for the 10-, 100-, and 1,000-pair families;
  canonical strong CC, BC, and BB power remained 200/200. Those reused results
  are design evidence only and cannot become confirmation.
- Generation 12 therefore freezes detector-family alpha at 0.020/0.020,
  retains the 0.09 publication gate, and precommits exactly 300 fixed-family
  replicates. The expected operating characteristic at true FWER 0.04 is
  approximately 96.6% probability of passing without changing the decision
  threshold after observing outcomes.
- The prior Gate-B PASS is demoted to development-only for this scientific
  generation because its alpha allocation differs. Publication readiness now
  requires a fresh 5,600-task Gate B and a fresh 300-task fixed-family
  supplement using contract-hash-derived seeds with proven zero overlap. The
  failed generation-11 artifacts and exact Kestrel accounting remain immutable.
  No applied, recovery, or publication stage may resume before both new
  reducers independently pass.
- Development evidence is preserved under
  `run3/alpha_0020_development_20260822T140500Z`: fixed-family payload SHA-256
  `c04fb41e8ca45341de2a4436690e9e1024faec29be5ec5843206e0f1f2fe6d7c`
  and canonical-power payload SHA-256
  `288987d16a51fbd196abd2f99573ec7acaf05ec9e83d1d164b45dc3fcc294a02`.
  The fresh confirmation package must not consume these payloads as results.
- The paired RFM scientific contract is published at
  `fb8b57f180ac24aba9ed6df01726cb1f380ee65f`. BSM pins that exact revision and
  regenerated lock SHA-256. The development-only rejected-history fixture was
  rebound to the new lock and its seed ledger regenerated from the fixture's
  new contract identity; the former rejected commit and lock hash remain
  recorded in the fixture header. The complete BSM suite passes (131 tests).

## SESSION STATE — 2026-08-18 — Prospective Gate-B BB factorial correction

- The completed development Gate B passed all five null regimes and recovered
  200/200 strong continuous--continuous and 200/200 strong binary--continuous
  interactions, but canonical strong binary--binary recovery was 103/200
  (power 0.515; one-sided 95% Wilson lower 0.4570604894), below the frozen 0.80
  gate. Downstream execution remains blocked.
- The root cause is a detector mismatch: after additive conditioning, the
  planted binary--binary signal is a parity contrast with zero population
  marginal split gain. The correction retains TreeSHAP for continuous--
  continuous and binary--continuous pairs and assigns binary--binary pairs to
  a studentized HC3 saturated 2x2 factorial contrast.
- The G11-v10 contract freezes separate maxT families at alpha 0.025 and 0.025,
  requires all four BB cells with at least two rows per cell, and increments
  the scientific artifact schema to 3. Detector identity is persisted and
  hashed for every pair; old score artifacts cannot be reduced as corrected
  evidence.
- The prior 5,600-task run remains immutable development/ablation evidence. A
  fresh 5,600-task Gate-B confirmation on new contract-derived seeds is
  mandatory after clean-code validation and a separate AU review. No job is
  submitted by this implementation slice, and no later stage may start before
  that confirmation passes.
- Manuscript method wording and the complete internal amendment are maintained
  in the paired manuscript worktree. All numerical results remain provisional
  until corrected Gate B and its dependent stages are regenerated.

## SESSION STATE — 2026-08-16 — Binary-only nonlinear terminalization

- Canonical `sacct --array` review found seven Gate-B null tasks in
  `FAILED/1:0`: `4040`, `4054`, `4124`, `4220`, `4304`, `4466`, and `4467`.
  Each failed after empirical screening legitimately retained only binary
  first-order terms; the generic nonlinear stage then rejected the resulting
  empty supported-transform family. The controller and all Gate-B/downstream
  jobs were stopped, and failure evidence was preserved under the campaign
  control root.
- The 4,921 exact completed worker results remain scientifically valid and
  immutable. The continuation must submit only the 679 missing tasks, must not
  reuse the seven failed jobs' telemetry, and must retain the corrected
  undirected-pair reducer from the preceding review finding.
- The adapter now scopes a reviewed `empty_candidate_family` terminal result
  to this exact valid condition, revalidates the candidate family, training
  rows, component columns, and active-component variance, and restores the
  pinned generic function after each worker call. Unrelated errors still
  propagate. A matching generic RFM correction is maintained separately for
  future source identities; this continuation keeps the accepted worker RFM
  identity and changes only the content-addressed adapter.
- Focused adapter regressions and the complete BSM suite pass (123 tests).
  Clean deployment, exact failed/cancelled AU accounting, continuation-cache
  audit, missing-task submission, and live completion remain required.

## SESSION STATE — 2026-08-16 — Undirected interaction recovery bookkeeping

- A live read-only audit of the first 1,817 completed Gate-B records found
  valid, hash-bound scientific artifacts but an order-sensitive derived metric:
  retained interaction `HTL:HEFA` was compared literally with truth
  `HEFA:HTL`. This mislabeled 45 of 87 completed `strong_cc` replicates as
  misses/false positives even though the planted undirected pair was retained.
- Only the derived terminal bookkeeping is affected. Predictions, model
  freezes, observed/null scores, seeds, schedules, truth ledgers, retained
  support, and comparator outputs remain valid and reusable. The existing
  Gate-B audit/reducer and all dependent fixed-family jobs were held; valid
  Gate-B workers remain eligible to finish.
- The adapter now canonicalizes undirected interaction IDs before terminal
  support metrics and recomputes Gate-B false-selection/power events from raw
  truth/retained IDs rather than cached booleans. The publication compiler
  likewise rebuilds every recovery metric from raw support IDs, so completed
  immutable records need no scientific rerun.
- Acceptance requires the reversed-order regression to produce one discovery
  and zero false pairs, malformed IDs to fail closed, publication CSVs to
  ignore deliberately stale cached metrics, all focused recovery/campaign/
  final-execution/publication tests to pass, and a provenance-preserving
  continuation reducer to consume the original worker result hashes.

## SESSION STATE — 2026-08-15 — Cache snapshot worker-count repair

- The independent continuation audit found that cached resolution score
  snapshots freeze the 65-CPU worker count, while login-node cache validation
  reconstructed the contract from its one-CPU environment. Promotion therefore
  failed closed before copying outputs or submitting a job.
- Interaction-spec construction now accepts an explicit worker count for
  offline identity reconstruction. Cache promotion requires source and target
  `worker_resources` to match and uses the manifest's requested CPU count;
  ordinary workers retain the unchanged `SLURM_CPUS_PER_TASK` behavior.
- Snapshot verification is loaded from the immutable scientific RFM checkout,
  not the separately versioned controller checkout. This preserves the exact
  implementation-tree and lock hashes that created the cached scores while
  still using the newer controller only for orchestration.
- The promotion regression deliberately makes the login environment differ
  from the manifest and proves all 20 cached shards validate and complete with
  zero scientific work units. Scientific methods and score bytes are unchanged.

## SESSION STATE — 2026-08-11 — G11-HPC-S1 paired sync note (uncommitted)

- This BSM worktree remains dispatch-blocked for G11:
  `scheduler_submission_permitted: false`.
- Historical 16,000-AU/15,152-AU package claim is superseded. The live paired
  full provisional package reports 175,449 requested AUs before pilot
  telemetry; production remains locked until pilot-selected resources produce
  a regenerated envelope that fits the same-day remaining allocation.
- Paired RFM slice `/Users/dhetting/src/rfm-pipeline-g11-integration`
  generated a content-addressed NO-SUBMIT Kestrel package only. No BSM
  execution, artifact regeneration, `sbatch`, commit, or push occurred here.
- The BSM pre-HPC suite does not require five future authoritative release
  figure PDFs. `scripts/reproduce_artifacts.py --validate-only` retains the
  strict later-release check for all five PDFs, including missing and
  truncated-file rejection. No stale figures were generated or promoted.

## SESSION STATE — 2026-08-12 — G11 integration implementation (NO-SUBMIT)

- Current branch: `g11-integration-reconcile`; changes are uncommitted and no
  scheduler command, commit, or push occurred.
- `configs/g11_campaign_contract.toml` records the publication G11-v9 contract,
  including the 6,400-record null/strong/stress inventory, 158 continuous plus
  two binary predictors, B-screen=3,199, initial B-interaction=999, 20
  resolution records, fixed-family supplement, comparators, retry policy, and
  2,000 bootstrap draws.
- `scripts/g11_campaign_adapter.py` executes manifest-bound resolution,
  Gate-B/fixed-family, Gate-C, and applied stages through the pinned RFM path.
  It validates the phase authorization and resource freeze and emits
  content-addressed terminal/scientific artifacts. Gate C reuses Gate-B strong
  records and adds only the four stress regimes.
- `scripts/prepare_g11_applied_data.py` prepares the 30,000-row split with a
  sealed 1,500-row holdout and records 160 inputs (158 continuous, two binary)
  plus 23,495 outputs. Adaptive jobs receive training bytes only; the existing
  holdout is not described as previously contaminated. The preparer rejects
  X/Y/assignment row-order drift and non-0/1 scenario values, and the prepared
  manifest binds the exact preparer bytes.
- Legacy publication submission entry points and scheduler defaults are
  fail-closed against G11 so they cannot silently run the superseded B=201/
  B=31/100-bootstrap route. The old method contract is preserved under
  `configs/rejected_history/`.
- Validation: the complete BSM test suite passes (`75 passed`); Ruff format and
  lint pass on the G11 adapter, splitter, driver, and tests; guarded shell
  entrypoints pass `bash -n`; `git diff --check` passes.
- Paired hashes in RFM `configs/hpc/g11_kestrel_campaign.yml` currently match
  the adapter, recovery driver, DGP contract, and applied config bytes. The
  adapter requires comparators only for strong/stress recovery records and
  validates schema-v2 run/source/lock/inventory-bound authorization before
  every scientific operation.
- Still blocked before any HPC submission: commit and install exact clean RFM/
  BSM checkouts, prepare and hash the real applied-data manifest, record
  same-day allocation/storage/inode evidence, run exact `sbatch --test-only`,
  and obtain separate user authorization for scheduler submission.

## SESSION STATE — 2026-08-14 — final execution/release closure (uncommitted)

- The accepted sizing pilot `g11-pilot-final-sizing-20260814f` completed;
  none of the changes in this section has altered its source-bound RFM commit,
  submitted another scientific phase, or changed its evidence root.
- `scripts/g11_final_execution.py` now provides one restartable controller for
  accepted-pilot accounting, development resolution, fresh confirmatory
  packaging, 25,000-AU certification, Gate B, Gate P, Gate C, and bounded
  publication compilation. Every submission requires literal `--execute`;
  same-day preflight occurs immediately before submission; existing completion
  records, reducer bytes, and budget certificates are revalidated on restart.
- `advance` and `watch` accept a fresh invocation-level `--remaining-au` so a
  later phase never depends on the initialization-day allocation snapshot. A
  changed `aus_report` value stops before preflight; restarting with the fresh
  integer continues the same immutable campaign without resubmitting work.
- Phase submission writes and fsyncs an append-only scheduler journal after
  every `sbatch` response so a partial submission cannot lose returned job IDs
  or silently resubmit. Exact terminal acceptance requires `COMPLETED/0:0` for
  every non-array job and every expected array task; an optional synthetic
  array-parent row may not substitute for task coverage.
- Whole-campaign accounting now reserves 5 AUs for a final one-CPU,
  30-minute, 8-GB `shared` publication-compilation job. That job creates the
  compact publication bundle on Kestrel while all immutable package/result
  paths remain available; all subsequent work is local and consumes no AUs.

## SESSION STATE — 2026-08-15 — fixed-family publication amendment (uncommitted)

- The successfully completed accepted pilot is
  `g11-pilot-final-sizing-20260814f`; its 63 steps and original 1,000-count
  pilot-base contract remain immutable evidence. Prior rejected attempts cost
  exactly 192.55 AUs, and the accepted pilot cost 90.36944444444445 AUs.
- Before confirmatory execution, one hashed amendment changes only
  `fixed_family_replicates` from 1,000 to 200. The five primary null regimes
  remain at 1,000 each. The 10/100/1,000-pair nested supplement uses all 200
  precommitted identities, one-sided 95% Wilson upper-bound \<=0.09, and largest
  passing event count 11. No interim test, optional stopping, or incremental
  standby extension is permitted.
- The final package and budget certificate bind the amendment hash. Package
  creation replaces completed pilot/resolution estimates with exact observed
  AUs, applies the shared 20% reserve only to unexecuted work, and remains
  fail-closed above 25,000 AUs after prior rejected attempts and the bounded
  5-AU publication job. The retained B=999 projection reserves 24,301 AUs for
  remaining confirmatory work and leaves at most 411.0805555555562 AUs for
  actual resolution. The B=1,998 branch requests 48,499 AUs for remaining work
  and is therefore an unconditional pre-confirmatory stop. No scheduler job was
  submitted by this amendment work.
- Scaling the accepted 6,342-second interaction pilot by the resolution's
  measured pair-draw ratio forecasts about 223 AUs including reducer allowance,
  below the 411.08-AU B=999 limit; admission still uses exact post-resolution
  `sacct`, never that forecast.
- Each confirmatory phase now receives an immutable rolling budget guard that
  substitutes observed AUs for completed phases and requires the current
  phase's complete scheduler-requested walltime charge to fit under the cap.
  This prevents an overrun in one phase from leaking into a later submission.
- `scripts/build_g11_publication_artifacts.py` compiles all publication tables,
  23,495-output ledgers, coefficient products, 11 figure-data surfaces,
  generated manuscript values/tables, and provenance. Every reducer artifact
  and raw recovery terminal record is hash-bound to its packaged reducer and
  worker result before use.
- `scripts/audit_g11_publication_artifacts.py`,
  `scripts/install_g11_manuscript_artifacts.py`, and
  `scripts/finalize_g11_manuscript_release.py` provide an independent audit,
  atomic manuscript installation, deterministic figure build, and local
  `manuscript.pdf`/`coverpage.pdf` release gate. The terminal local token is
  `JDS_RELEASE_BUILD_PASS`.
- Full operator commands, acceptance criteria, transfer instructions, and the
  zero-AU local finish are recorded in `docs/G11_FINAL_EXECUTION.md`.
- Validation currently passes: all 105 BSM repository tests, scoped Ruff
  lint/format, direct CLI entry-point smoke checks, all 266 manuscript
  repository tests, and successful 21-page manuscript plus one-page cover
  compilation with no unresolved citations or references.
  The release audit independently recomputes the raw-scale coefficient and
  intercept algebra from the frozen model, and the final artifact submission
  fsyncs its returned scheduler ID before writing the submission record. Final
  local success also requires deterministic, validated JDS source and
  publication-supplement archives. Final scientific values remain generated
  surfaces until the confirmatory campaign completes.
- The local reproduction layer now applies two tested text-only corrections
  after the pilot-bound RFM renderer returns: the per-output CDF uses a compact
  in-frame maximum annotation, and the module heatmap is titled `Interaction endpoint counts`. Plot geometry and scientific values are unchanged, and no
  RFM source/pin used by the active pilot was modified.
- A true local release rehearsal also passes using a production-sized synthetic
  23,495-output bundle and the real compiler, figure renderer, independent
  numerical audit, installer, LaTeX toolchain, and archive writer. It produced
  44 checksummed artifacts, 11 SVGs, a 20-page synthetic-result manuscript, a
  one-page cover, and internally valid source/supplement ZIPs. The supplement
  now carries its own reproduction guide and license so it remains usable when
  separated from this repository.
- Live no-submit initialization exposed and closed two related environment
  hazards: the older BSM Pixi environment imported a stale installed RFM
  package, while the accepted scientific checkout intentionally predates the
  non-scientific pilot-accounting schedule-hash repair. The final controller
  now binds and hashes a separate `77a8e80` controller checkout while package
  source hashing and worker execution remain bound to the accepted `fd12fd5`
  scientific checkout. The runbook supplies that separation explicitly and
  the controller rejects a changed or wrongly imported controller module. The
  controller import path changes only in-process; every real Slurm submission
  strips `PYTHONPATH`/`PYTHONHOME`, so workers remain bound to the accepted
  scientific checkout. The manifest records clean Git revisions and complete
  RFM-controller/BSM-script tree hashes. Recomputed pilot evidence must match
  the accepted freeze in every scientific and numeric field; the sole
  permitted difference is newer explanatory `accounting_rule` prose, in which
  case the accepted self-hashed freeze bytes remain authoritative. The
  publication job uses
  the scientific RFM Python environment rather than the stale BSM environment.
  A partial multi-`sbatch` failure cancels every returned job ID and writes a
  self-hashed abort record. Because `nationalpfa` is valid in Slurm but omitted
  from `aus_report` for this non-lead user, live smoke uses the configured
  25,000 ceiling while the separate campaign certificate and rolling guards
  subtract all sunk and observed AUs. No scheduler job was submitted by either
  rejected initialization.

## SESSION STATE — 2026-08-15 — final-submission adversarial hardening

- Scientific submission remains frozen while the exact final controller bytes
  complete local and Kestrel no-submit validation. No final scientific job was
  submitted by this review.
- Development package generation is now structurally resolution-only. Its
  immutable admission guard requires the complete resolution walltime request,
  the accepted pilot, all rejected attempts, the 5-AU publication reserve, and
  the full B=999 confirmatory reserve to fit below 25,000 AUs before resolution
  can be authorized.
- The 65-CPU, 6-GiB resolution workers retain their pilot-sized 2:43:33
  walltime but use Kestrel's 104-core `shared` nodes. This removes billing for
  the unused 39 cores and makes the full resolution walltime request fit while
  leaving the scientific computation and resource envelope unchanged.
- Generated worker/audit/reducer scripts use package-local precreated Slurm log
  directories, clear Python import overrides, and bind their SHA-256 values in
  the submission plan. Same-day preflight and authorization bind the exact plan
  hash, and the submitter rehashes every script before its first `sbatch`.
- Post-pilot arrays are accepted and charged from every expected task row;
  missing, extra, failed, or nonzero-exit tasks stop downstream work. Shared
  jobs use their request-bound node-equivalent fraction rather than charging a
  full node solely because `AllocNodes` reports one.
- The publication compiler has its own same-day `sbatch --test-only` evidence,
  binds that evidence into its submission record, journals the returned job ID,
  and refuses stale or partial evidence on restart.
- Focused controller tests pass (28), the complete BSM suite passes (112), and
  all 112 tests also pass under the scientific RFM Pixi interpreter with
  `PYTHONPATH`/`PYTHONHOME` removed and user-site imports disabled. The complete
  RFM repository gate passed immediately before the later isolated site-utility
  environment repair; fresh Kestrel replay remains required before submission.
- A prospective B=999 confirmatory-package audit then exposed a blocking
  mismatch between the generic package gate and the later exact budget
  certificate: the generic gate re-reserved estimates for already completed
  pilot and resolution work and rejected a 25,215-AU aggregate before the
  24,929.82-AU exact certificate could be built. No scheduler job was
  submitted. Confirmatory admission now uses immutable observed AUs for
  completed work, reserves 20% only on unexecuted confirmatory stages, and
  counts the separate 5-AU postprocessing reserve. The controller supplies the
  accepted-pilot, rejected-attempt, and observed-development charges from the
  resource freeze and retained accounting. Full validation plus fresh B=999
  acceptance, B=1,998 rejection, and every-script Kestrel `sbatch --test-only`
  remain mandatory before the execution flag may be used.
- The first same-day preflight with the isolated scientific interpreter then
  exposed an environment boundary in NREL's `aus_report`: inherited
  `PYTHONNOUSERSITE=1` hid the site utility's own user-site `jwt` dependency.
  No phase authorization, submission record, journal, or scheduler job was
  created. RFM controller `77a8e80` removes that variable only for the external
  allocation probe; every scientific worker retains clean import isolation.
  The complete G11 HPC package test file, focused BSM controller tests, Ruff,
  and diff checks pass; a new immutable root and same-day replay are required.

## SESSION STATE — 2026-08-15 — corrected final scientific campaign submitted

- Runs `g11-final-manuscript-20260815h` and `...j` are rejected diagnostics,
  not scientific evidence. Run `h` exposed a wrong manifest authorization
  path; run `j` exposed the BSM verifier's stale exact authorization schema.
  Both failed before scientific computation. Their immutable evidence records
  0.09161324786324784 and 0.036458333333333336 AU, respectively, bringing
  cumulative rejected-attempt usage to 192.6780715811966 AUs. All unfinished
  work and dependencies were canceled; no failed-run telemetry is eligible for
  resource selection.
- RFM `7cc6f09` generates the canonical phase authorization path and BSM
  `b3b2377` checks every manifest path before submission and uses canonical
  Slurm array task identifiers. BSM `d7d5f32` requires and validates the
  plan-bound authorization field that failed run `j`; RFM `7bc07b1` pins that
  exact BSM scientific checkout and driver digest. The complete BSM suite
  passes (115), the complete G11 RFM package test passes, and the failed live
  authorization replays successfully through the corrected verifier on
  Kestrel.
- Fresh immutable run `g11-final-manuscript-20260815k` binds scientific RFM
  `fd12fd579d8743bdc4acd00e1dac217cbfc56e84`, controller RFM
  `7bc07b1f1f848b2eb8ab4ad429d79909f85aeb0e`, and BSM scientific/controller
  runtime `d7d5f32b586a5b9d06182666aee24d0e94160ffd`; all deployed checkouts are
  clean.
- No-submit audit `8f64b94be45ed57651a4b149a54096b6db12aa9e5de8f4585a48722d166f3ecd`
  validates all 6,689 manifest records, passes `sbatch --test-only` for all 49
  development and confirmatory scripts, admits B=999 at a
  24,929.94762286325-AU hard maximum, rejects B=1,998, and preserves
  fixed-family 200 plus five primary null regimes of 1,000. The exact live
  development authorization then passed all 20 scientific-verifier replays
  before submission; audit SHA-256
  `90763ffdff43f2091520232115b89fd5f864dd8d5148fad5b22b55b78a8a2634`.
- Development worker array `16278789`, audit `16278790`, and reducer `16278791`
  are submitted with reviewed `afterany`/`afterok` dependencies. Lightweight
  controller PID 1133692 monitors at 300-second cadence and may advance only
  through exact completion evidence, scientific gates, same-day preflights,
  and rolling allocation guards. Scientific failures are never retried
  automatically; a 30-minute heartbeat monitors the campaign.

## SESSION STATE — 2026-08-15 — resolution terminalization repair and cache continuation

- Final run `g11-final-manuscript-20260815k` is stopped and rejected after all
  20 resolution workers failed at the same post-compute adapter line. The
  controller is dead, the reducer was canceled before execution, and the queue
  is empty. A 250-file, 125-MiB evidence bundle verifies under SHA-256 at
  `/projects/bsm/g11_failure_evidence/g11-final-manuscript-20260815k_scientific_adapter_failure_20260815T182652`.
- Exact failed-attempt cost is 118.40357905982906 AUs; cumulative rejected
  attempts are 311.08165064102566 AUs. Failed-job telemetry is diagnostic only.
- The adapter now uses `payload_sha256` in resolution, fixed-family, and applied
  interaction outputs. Resolution decision materialization is separated from
  score generation so completed self-verifying score blocks can be promoted.
- The final controller accepts one content-bound resolution failure bundle at
  initialization. It holds target outputs empty through preflight, promotes
  only exact science-identity matches after authorization, verifies each score
  block against the target contract, and writes standard result/success bytes.
  Fresh workers therefore validate and exit without repeating scoring; audit
  and reduction remain mandatory.
- Local acceptance: 51 focused tests pass; the complete BSM suite passes in
  both the BSM environment and the exact RFM integration environment; Ruff and
  diff checks pass on modified surfaces. Remaining work is content-addressed
  commit/push, RFM config repin, clean Kestrel deployment, independent no-submit
  audit, cache promotion, and continuation submission/monitoring.

## SESSION STATE — 2026-08-19 — downstream continuation readiness repair

- The paired RFM generator is pinned to fully validated commit
  `a3bcbaa389bc7a708ca644f4e95f9b9268407e21`. Its downstream-only package mode omits Gate B,
  starts at the approved 200-replicate fixed-family supplement, binds every phase authorization
  to the exact submission-plan hash, and admits the remaining workflow using exact completed AUs
  plus the shared 20% reserve on only unexecuted work and the five-AU publication reserve.
- `g11_campaign_workflow.py` can carry a passing 5,600-record Gate-B decision across the signed
  publication amendment only when `fixed_family_replicates: 1000 -> 200` is the sole contract
  change. The adoption record binds the old/new contract identities, source decision file hash,
  amendment hash, exact record count, and its own canonical hash. Fixed-family is now a distinct
  authorization phase whose only prerequisite is that adoption record.
- Updating the RFM pin exposed an environment defect: the rejected-history G0/B contract bound
  its generic commit and lock checksum to the former live Pixi environment. The contract, lock,
  and deterministic seed ledger were advanced together following the repository's established
  pin-update convention; this changes no active Gate-B seed, schedule, model, result, or submitted
  job. The regenerated Pixi lock installs the exact repaired RFM commit.
- Validation passes: scoped Ruff format/lint, lock installation, exact installed-package smoke,
  and all 126 BSM tests. No downstream scheduler job was submitted.

## SESSION STATE — 2026-08-22 — generation-12 prerequisite adoption

- The prospective generation-12 confirmation reuses only the successful pilot
  resource selections and the accepted B=999 resolution. It does not reuse the
  superseded Gate-B or failed fixed-family scientific results.
- `adopt-generation-12-prerequisites` now binds the exact old freeze, resolution
  bytes, and development completion into four new self-hashed records under the
  current RFM source, lock, base-contract, and fresh run identity. The exact
  telemetry and selected resource profiles remain byte-equivalent and are
  separately hashed; `resource_decision_changed` is false.
- Fresh package admission must prove 5,600 Gate-B plus 300 fixed-family task
  identities, zero seed overlap with the prior 5,800 tasks, native RFM package
  validation, and both-host Linux runtime preflight before scientific launch.
