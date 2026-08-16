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

### REVIEW-0013 — Campaign contingency differed between controller and package config

- Status: fixed
- Severity: high
- Category: correctness
- Disposition: blocker
- Source: fresh continuation prospective-package audit
- Evidence: B=999 stopped before submission at 25,048.4 AUs because the outer controller used the user-authorized 25,100-AU contingency while the canonical package config still enforced 25,000 AUs
- Affected files: `configs/hpc/g11_kestrel_campaign.yml`, `tests/alignment/test_G11_HPC_S1_kestrel_package.py`
- Required action: align the canonical admission ceiling and its regression to 25,100 AUs; preserve all scientific settings and reserves
- Blocks merge: no
- Destination: resolved in the continuation controller repin
- Notes: expected post-cache actual projection remains below 25,000 AUs; no scheduler job was submitted by the failed audit

### REVIEW-0014 — Cache validation inherited the login-node CPU count

- Status: fixed
- Severity: high
- Category: correctness
- Disposition: blocker
- Source: independent cache compatibility audit
- Evidence: cached Slurm snapshots bind 65 workers, but the initial offline validator reconstructed one worker from the login-node environment and correctly rejected the checksum
- Affected files: BSM cache promotion; exact BSM path/hash pins in `configs/hpc/g11_kestrel_campaign.yml`
- Required action: reconstruct from manifest-bound worker resources, test with a deliberately different login environment, and repin exact BSM bytes
- Blocks merge: no
- Destination: resolved by BSM `dfa8599` and this controller repin
- Notes: score artifacts and scientific settings are unchanged; no job was submitted

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

### REVIEW-0005 — Round 15 future-wave RF fallback and guidance-table drift

- Status: deferred
- Severity: medium
- Category: reproducibility
- Disposition: required_follow_up
- Source: 2026-06-07 round-15 adversarial audit at commit 4183b67
- Evidence: `plot_sensitivity_results.py` reused `wave1_rf_quality.pkl` when a future `<prefix>_rf_quality.pkl` was absent; manuscript `jds_bsm_v22.tex:725-726` still lists dead `LASSO $\alpha$ percentile` guidance and stale BSM-coupled runtime guidance.
- Affected files: `scripts/plot_sensitivity_results.py`; external manuscript `jds_bsm_v22.tex`; `docs/manuscripts/manuscript_impact_log.md`
- Required action: code fallback fixed; add a regression test for missing future-wave RF pickles and edit the manuscript guidance table before submission.
- Blocks merge: no
- Destination: manuscript edit pass plus future test-hardening slice
- Notes: wave12 numeric replacements were recomputed from `artifacts/sensitivity/wave12_combined_clean.csv` during the audit.

### REVIEW-0006 — Round 16 sensitivity interaction-threshold terminology drift

- Status: deferred
- Severity: medium
- Category: manuscript_alignment
- Disposition: required_follow_up
- Source: 2026-06-07 round-16 adversarial audit at commit 40ffe3b
- Evidence: external manuscript `jds_bsm_v22.tex:623` says `Interaction null-quantile threshold` baseline `0.995` / swept `[TBD]`, but `artifacts/sensitivity/wave12_combined_clean.csv` contains `stages.interaction_discovery.p_threshold` levels `(0.01, 0.05, 0.10, 0.20)` and sensitivity code models that p-threshold column.
- Affected files: external manuscript `jds_bsm_v22.tex`; `docs/manuscripts/manuscript_impact_log.md`; `tests/test_sensitivity_study.py`
- Required action: edit manuscript sensitivity-study parameter table to use `Interaction p-threshold` or explicitly split case-study null-quantile from synthetic sensitivity p-threshold.
- Blocks merge: no
- Destination: manuscript edit pass
- Notes: mechanical guard added to pin the p-threshold sweep in the config-options regression test.

### REVIEW-0007 — Round 17 future-wave LASSO-grid and scratch-path hardening

- Status: deferred
- Severity: medium
- Category: reproducibility
- Disposition: required_follow_up
- Source: 2026-06-07 round-17 adversarial audit at commit 39c3f19
- Evidence: `tests/test_sensitivity_study.py::test_config_sweep_options_match_manuscript_table4_values` did not assert the now-live `stages.sparse_selection.lasso_alpha_grid_size` sweep `(20, 40, 80, 160)`; `scripts/run_sensitivity_job.py` wrote generated synthetic Parquet/config files to the host default temp directory; external manuscript `jds_bsm_v22.tex:579` contains `two family]ies`.
- Affected files: `tests/test_sensitivity_study.py`; `scripts/run_sensitivity_job.py`; external manuscript `jds_bsm_v22.tex`; `docs/manuscripts/manuscript_impact_log.md`
- Required action: code/test hardening completed; edit manuscript typo before submission; later replace the three Chrome/tempfile SVG→PDF helpers with a repo-local, configurable converter.
- Blocks merge: no
- Destination: manuscript edit pass plus future reproducibility-hardening slice
- Notes: wave12 still has constant LASSO grid size 40; do not cite `(20, 40, 80, 160)` as observed variation for current wave12 results.

### REVIEW-0008 — Round 18 runtime-doc examples call removed attributes

- Status: **fixed (round 18 commit)**
- Severity: medium
- Category: documentation_reproducibility
- Disposition: closed
- Source: 2026-06-07 round-18 adversarial audit at commit 1ebc4d6
- Evidence: `pixi run python` confirmed `resolve_manuscript_runtime(Path.cwd())` returns `ManuscriptRuntimeContext demo False` for `hasattr(rt, "x_train")`, then `AttributeError: 'ManuscriptRuntimeContext' object has no attribute 'x_train'`; docs still instruct new users to print `rt.x_train`, `rt.y_train`, `rt.x_train_path`, `rt.y_train_path`, `rt.x_holdout`, `rt.y_holdout`, and `rt.artifact_root`.
- Affected files: `docs/configuration_reference.md`; `docs/setup_and_first_run.md`
- Resolution: replaced all 4 broken snippets across the two files with supported `rt.mode`, `rt.output_root`, `rt.repo_root`, `rt.local_override_used`, `rt.unresolved_placeholders`, and `rt.artifact_paths` printouts. Expected-output block in setup_and_first_run.md updated to match.
- Blocks merge: no
- Destination: JOSS/new-user documentation cleanup before release
- Notes: deferred snippet smoke test still open as future work (would catch the next ManuscriptRuntimeContext shape drift in CI).

### REVIEW-0009 — Round 19 docs still bypass Pixi for demo runs

- Status: **fixed (round 19 commit)**
- Severity: medium
- Category: documentation_reproducibility
- Disposition: closed
- Source: 2026-06-07 round-19 adversarial audit at commit 181bc83
- Evidence: repo onboarding says users do not need pre-installed Python and repo policy requires Pixi, but 10 markdown snippets still run `PYTHONPATH=src python examples/end_to_end_reproducibility.py` instead of `pixi run python ...`.
- Affected files: `docs/quickstart.md`; `docs/reproducibility_example.md`; `docs/setup_and_first_run.md`; `docs/configuration_reference.md`
- Resolution: bulk `sed` replaced all 12 occurrences (auditor undercounted at 10) of `PYTHONPATH=src python` with `pixi run python` across the four files.
- Blocks merge: no
- Destination: JOSS/new-user documentation cleanup before release
- Notes: round 20 found broader command-policy drift still open in other new-user troubleshooting snippets; see REVIEW-0010.

### REVIEW-0010 — Round 20 new-user docs still contain non-policy commands

- Status: **fixed (round 20 commit)**
- Severity: medium
- Category: documentation_reproducibility
- Disposition: closed
- Source: 2026-06-07 round-20 adversarial audit at commit 5cfe5c0
- Evidence: `docs/troubleshooting.md:91-122,199-200` still teaches `PYTHONPATH=src pixi run python`; `docs/setup_and_first_run.md:343-359` recommends `pixi shell` and bare `python`; `docs/configuration_reference.md:335,362` uses bare `python -c`. These conflict with the Pixi-first setup promise and repo policy, and there is still no docs-snippet smoke test to catch command drift.
- Affected files: `docs/troubleshooting.md`; `docs/setup_and_first_run.md`; `docs/configuration_reference.md`; `docs/DOCUMENTATION_IMPROVEMENT_PLAN.md`
- Resolution: bulk-replaced `PYTHONPATH=src pixi run python` → `pixi run python` in troubleshooting (3 occurrences + cause/comment text); converted bare `python -c` → `pixi run python -c` in setup_and_first_run + configuration_reference (3 snippets); replaced `pixi shell` recommendation with explicit note that repo policy prefers `pixi run`; corrected `DOCUMENTATION_IMPROVEMENT_PLAN.md` troubleshooting bullet.
- Blocks merge: no
- Destination: JOSS/new-user documentation cleanup before release
- Notes: docs-snippet smoke test still open as future work — would catch the next bare-python or `pixi shell` regression in CI.

### REVIEW-0011 — Round 21 setup doc retained non-policy shell guidance

- Status: fixed
- Severity: medium
- Category: documentation_reproducibility
- Disposition: closed
- Source: 2026-06-07 round-21 adversarial audit at commit 72cde47
- Evidence: after the round-20 cleanup, `docs/setup_and_first_run.md:336-348` still said to ensure `PYTHONPATH` was set and kept a discouraged `pixi shell && python ...` snippet in the ModuleNotFoundError troubleshooting section. This conflicted with Pixi-first onboarding and made command-policy drift easy to copy.
- Affected files: `docs/setup_and_first_run.md`
- Resolution: replaced the stale troubleshooting text with a single repo-root `pixi run python ...` instruction and removed the persistent-shell snippet.
- Blocks merge: no
- Destination: JOSS/new-user documentation cleanup before release
- Notes: broader docs-snippet smoke coverage remains deferred; D1 SVG→PDF Chrome/tempfile helper rewrite remains open.

______________________________________________________________________

## 2026-06-07 — Round 25: orchestrator parity + provenance labels

- **Severity:** HIGH (rfm).
- **Evidence:**
  - rfm `tools/run_hpc_workflow.py` did not honor the
    `--dry-run --generate-only` SSH-skip contract that the sibling
    bsm orchestrator now enforces (round 24 MED#9). Local validation
    against the rfm entrypoint still required SSH credentials.
  - rfm `docs/RUNNING_MANUSCRIPT_REPRODUCTION.md` lines 40-58
    documented `pixi run hpc-workflow -- --config configs/hpc/kestrel_workflow_orchestration.yml` and
    `kestrel_cpu_scale_2_smoke.yml`, but no such configs are
    committed to this repo (the canonical examples live in
    `bsm-public-rf/configs/hpc/`). A new user following the doc
    would fail at first command.
  - rfm `src/rfm_pipeline/distributed/slurm_array_runner.py:81`
    SLURM script template still labels itself "BSM Manuscript
    Pipeline" even though the package was renamed rfm-pipeline.
- **CLOSED (round 25, rfm-pipeline pending commit):**
  - `tools/run_hpc_workflow.py` short-circuits on
    `--dry-run --generate-only`: prints `>>> [skip ssh: ...] <cmd>`
    and continues without invoking SSH, mirroring bsm.
  - `docs/RUNNING_MANUSCRIPT_REPRODUCTION.md` HPC section now
    documents that orchestration configs and kestrel scripts are
    repo-specific, points at `bsm-public-rf` as the canonical
    reference, and uses `--generate-only --dry-run` as the smoke
    path (no nonexistent smoke config required).
  - `slurm_array_runner.py` SLURM script banner changed from
    "BSM Manuscript Pipeline" to "rfm-pipeline".
- **Deferred (round 25 audit, not implemented):**
  - hpc_workflow_config.build_status_command /
    build_collect_command point at bash scripts that live only in
    the consuming repo (bsm-public-rf). This is by design — the
    rfm-pipeline package is generic and expects the consumer to
    supply orchestration shell scripts — and is now documented in
    RUNNING_MANUSCRIPT_REPRODUCTION.md. No code change.
  - tools/hpc_bundle_manifest.py:466 hardcodes
    `hpc_shards_interaction_discovery` because the summarizer is
    designed for CPU/GPU scaling benchmarks (single-stage), not
    cascade runs. Comment added in round 24; no behavior change.
- **Tests:** 473 pass / 11 skipped.

______________________________________________________________________

## 2026-06-07 — Round 25 follow-up: deferred items addressed at root

- **Severity:** MED (lifted to closed).
- **Evidence:** the round-25 register flagged two items as
  "deferred by design"; per maintainer direction these are root-fix
  candidates and were closed without leaving behavior surprises.
- **CLOSED:**
  - `hpc_workflow_config.build_status_command` /
    `build_collect_command` no longer hardcode bash script paths.
    Two new optional fields on `HpcPathConfig`
    (`remote_status_script` = `scripts/kestrel/status_all_tests.sh`,
    `local_collect_script` = `scripts/kestrel/pull_hpc_artifacts_bundle.sh`)
    let consumer repos point at any script. `build_collect_command`
    now raises `FileNotFoundError` with a clear maintainer message
    when the configured script does not exist locally — earlier
    behavior silently constructed a `bash <missing>` command that
    only failed at execution time. Two new tests cover the
    missing-script error and the custom-path override.
  - `tools/hpc_bundle_manifest.py` summarizer is now cascade-aware.
    New `_discover_stages_in_run_dir` enumerates
    `hpc_shards_<stage>/` directories; `_summarize_target` accepts
    `stage=`; the implicit-tier path emits one summary row per
    discovered stage when the run is cascade (multi-stage). Legacy
    single-stage benchmark runs retain identical behavior and row
    labels. Four new tests cover discovery, per-stage isolation,
    and the legacy fallback.
- **Tests:** 479 pass / 11 skipped (was 473; +6 new tests).

______________________________________________________________________

## 2026-06-07 — Round 26: multi-tier cascade chaining + summarizer hardening

- **Severity:** HIGH (chaining bug).
- **Evidence:**
  - `tools/run_hpc_workflow.py` cascade chained the NEXT stage only
    to the LAST tier's reduce job id. When a stage group contained
    multiple tier invocations (CPU 2/10/1000 + optional GPU) each
    submitted its own reduce, but only the last id was captured and
    threaded into the next stage's `--depends-on-job-id`. Earlier
    tiers' reduces could still be running while the next stage's
    arrays started, racing on missing upstream artifacts.
  - `tools/hpc_bundle_manifest.py`: the `--target-specs-json` path
    skipped cascade stage discovery entirely, so study-package
    summaries hid non-interaction stage failures.
  - The implicit-tier path's suite_manifest fallback always pointed
    at `cpu_nodes_<tier>/hpc_scripts/manifest.jsonl` (legacy
    un-suffixed); cascade runs that wrote
    `cpu_nodes_<tier>_<stage>/hpc_scripts/manifest.jsonl`
    therefore reported `manifest_shards=0`.
  - Single-stage discovery fell back to the legacy `None` label —
    a mid-cascade run that had only emitted
    `hpc_shards_output_conditioning/` was summarized as
    `scripts_missing` against the wrong (interaction_discovery)
    stage.
  - `build_collect_command`'s new `FileNotFoundError` bubbled as a
    raw traceback in the CLI.
- **CLOSED (round 26, rfm-pipeline pending commit):**
  - New `parse_all_reduce_job_ids` helper in `hpc_cascade.py`.
    `inject_dependency_flag` now accepts `int | str | Sequence[int]`
    and renders multi-id upstreams as the SLURM `afterok` colon
    grammar (`123:456:789`). Empty sequence and mixed-type sequence
    raise (failing closed rather than silently dropping ids).
  - `hpc_submit.py` `--depends-on-job-id` parser accepts both bare
    ints and colon-separated lists, validating every segment.
  - `slurm_array_runner.write_scripts` / `generate_*_script` type
    hints widened to `int | str | None` (the existing f-string
    interpolation already handled str).
  - `tools/run_hpc_workflow.py` cascade loop captures one id per
    per-tier invocation (via `parse_reduce_job_id` on each captured
    stdout chunk) and chains the FULL id list into the next stage.
    `CascadeChainError` now fires if ANY tier in a stage group
    failed to emit a marker — partial chaining is unsafe.
  - `build_collect_command` `FileNotFoundError` caught at the CLI
    boundary and printed as `ERROR: ...` to stderr with exit code 2.
  - `tools/hpc_bundle_manifest.py`: new `_stages_to_summarize`
    (single-stage cascade now expands; only no-cascade falls to
    `[None]`); new `_stage_suffixed_manifest` (swaps
    `cpu_nodes_<tier>` → `cpu_nodes_<tier>_<stage>` when sibling
    exists, falls back otherwise). Both the `target-specs-json`
    and implicit-tier branches use the new helpers; the GPU
    summary follows the same expansion rule.
- **Tests:** 488 pass / 11 skipped (was 479; +9 new tests covering
  multi-id chain, empty/bad-type rejection, single-stage cascade
  expansion, stage-suffixed manifest lookup, and idempotency).

## REVIEW-0012 — G11 phased authorization could submit stale or partial science

- Date: 2026-08-12
- Severity: P0 / HPC-submission integrity
- Status: fixed in the current uncommitted integration branch
- Evidence:
  - preflight required current remaining AU to equal the original package
    allocation value, which becomes false as soon as an earlier tranche spends
    AU and makes later phases impossible to authorize honestly;
  - all confirmatory stages shared one broad `production` preflight rather than
    exact Gate-B, applied-production, and Gate-C script/resource inventories;
  - phased execution compared contract and inventory but omitted source and
    dependency-lock identity before invoking the scheduler;
  - phase authorization accepted prerequisite-shaped JSON without proving that
    it was the packaged reducer artifact protected by the packaged reducer
    `_SUCCESS.json`; and
  - a reducer depended only on a successful `afterany` audit. An audit can
    successfully report incomplete workers, so that dependency alone could
    release scientific reduction after a worker failure.
- Resolution:
  - tranche-specific same-day preflights now use current positive remaining AU
    and require it to exceed 125% of the selected tranche envelope;
  - phases are exactly `pilot`, `development`, `gate_b`, `gate_p`, and `gate_c`;
  - execution binds source, contract, lock, campaign inventory, current date,
    resource freeze, and authorization hashes before scheduler calls;
  - prerequisite decisions must be the exact packaged reducer outputs, be
    listed in content-addressed scientific artifacts, and have a valid packaged
    reducer result and success marker; and
  - reducers require `afterok` on all worker jobs plus the `afterany` audit,
    while audits remain available for failure diagnosis.
- Tests: new negative tests reject source/lock drift, stale inventory, forged
  prerequisites, and invalid phase evidence; focused HPC package suite and the
  complete `./test_repo.sh --check` gate pass.
- Blocks merge/submission: no after commit and clean-cluster preflight; the
  current dirty/uncommitted worktree and unresolved live Kestrel evidence still
  block scheduler submission.

## REVIEW-0013 — G11 resource classes and authorization were under-bound

- Date: 2026-08-12
- Severity: P0 / one-shot HPC execution integrity
- Status: fixed in the current uncommitted integration branch
- Evidence: null calibration unnecessarily ran all recovery comparators; one
  strong-recovery pilot resource request covered both 160x160x4 null jobs and
  2000x160x40 recovery jobs; a selected 1,998-draw schedule did not scale
  simulation walltime; the BSM driver was dynamically imported without a
  manifest hash; and written phase authorization omitted package/preflight/
  prerequisite identities that it had validated.
- Resolution: comparator artifacts are mandatory only for all strong/stress
  records; a three-profile exact-scale null pilot now freezes a separate Gate-B
  null array while the strong array retains recovery-pilot resources; the two
  arrays still feed one 5,600-record reducer; selected interaction draws scale
  both simulation resource classes; the recovery driver and applied-data
  preparer are content-bound; and
  schema-v2 authorization binds run, source, lock, inventory, preflight, and
  prerequisite hashes at write, submission, and worker execution boundaries.
- Validation: prospective failures were observed for comparator scope,
  1,998-draw scaling, driver identity, authorization replay, and resource-array
  separation before implementation; focused regressions and the complete HPC
  package suite pass after repair.
- Blocks merge/submission: unresolved live Kestrel/data/commit evidence remains
  unchanged; no scheduler command was executed.

## REVIEW-0014 — Demo runtime scratch accumulated on local workstations

- Date: 2026-08-12
- Severity: P0 / local execution availability
- Status: fixed in the current uncommitted integration branch
- Evidence: the demo runtime used `tempfile.mkdtemp` without an owner, leaving
  roughly 65 MiB after each construction. More than 800 directories exhausted
  the laptop filesystem. Context-owned temporary directories fixed ordinary
  Python calls, but notebook kernels could still be terminated before their
  finalizers ran and left six smaller remnants after a complete gate.
- Resolution: demo contexts now own a `TemporaryDirectory`; the notebook runner
  also assigns each kernel a scratch parent inside its own temporary execution
  directory, which is removed after the subprocess exits regardless of kernel
  finalization. Pytest now retains at most one failed temporary run and removes
  successful-run scratch, instead of retaining three package-heavy runs.
  Neither production paths nor HPC retention policy changed.
- Validation: both prospective cleanup regressions pass; a real manuscript
  notebook completed without increasing the residual demo-directory count; the
  six pre-fix notebook remnants were removed; zero demo directories remained;
  three pre-policy pytest runs totaling 3.0 GiB were removed; and local free
  space was 194 GiB after cleanup. The complete repository gate
  passed before the final notebook-boundary hardening, and its affected focused
  tests plus the real-notebook execution passed afterward.
- Blocks merge/submission: no; this is laptop scratch hygiene only.

## REVIEW-0015 — Pilot preflight and campaign admission envelope were incomplete

- Date: 2026-08-12
- Severity: P0 / one-shot HPC execution integrity
- Status: fixed in the current uncommitted integration branch
- Evidence: pilot executes the BSM Gate-B-null and recovery paths, but pilot
  preflight skipped every bound BSM executable/config hash. Full package
  generation also tolerated a nonempty destination. The reported 175,449-AU,
  9,528-GiB, 57,165-inode envelope covered primary attempts plus headroom but
  omitted the contract's one permitted infrastructure retry for each of 7,593
  workers, contrary to the manuscript handoff's requirement to budget every
  attempt and retry slot.
- Resolution: pilot preflight now validates the BSM adapter, recovery driver,
  DGP contract, and applied-data preparer; all package generation requires an
  empty destination; and whole-campaign plus tranche preflight envelopes
  include all retry worker resources/artifacts before applying 25% campaign
  headroom. The provisional worst-case pre-pilot bound is now 438,589 AUs,
  19,019 GiB, and 114,113 inodes. The no-retry planning estimate remains
  140,167 AUs, while primary requested walltime would charge at most 175,449
  AUs before post-pilot resource selection.
- Operational closure: a `live-smoke` command gathers same-day `aus_report`,
  clean Git/commit, Lustre project-quota, scratch-capacity, and exact pilot
  `sbatch --test-only` evidence into hashed records. Its scheduler command path
  is restricted to `sbatch --test-only`; it cannot submit work.
- Validation: prospective tests failed for stale full-package roots, stale BSM
  pilot bytes, missing live-smoke implementation, and unresolved allocation;
  the focused HPC package suite passes after repair.
- Blocks merge/submission: yes until the paired repos are committed, BSM pins
  that exact RFM commit, clean checkouts are installed on Kestrel, and the live
  smoke passes against the actual allocation/quota. No scheduler job ran.

### 2026-08-13 correction — worst-case retry reservation rejected

The 438,589-AU figure above is retained as audit history, not as the operative
budget. It assumes all 7,593 workers consume their complete requested walltime,
all 7,593 workers use their one permitted retry, and a further 25% campaign
factor is applied. That is a scheduler stress bound, not expected experiment
usage, and it made a 25,000-AU publication campaign impossible by construction.

The implemented policy now uses the minimum defensible control: live smoke is
zero-AU `sbatch --test-only`; the bounded pilot is checked as its own tranche;
accepted pilot telemetry freezes the lowest projected-AU profiles; and a final
package is rejected unless the whole projected campaign plus one shared 20%
overrun/retry reserve fits the fixed 25,000-AU ceiling. Individual retry
authorization remains limited to eligible infrastructure failures and the retry
package rebuilds the audit/reducer dependency chain. The deterministic
post-pilot test fixture projects 10,322 AUs including the completed pilot,
resolution, confirmatory production, and shared reserve.

## REVIEW-0016 — Kestrel project-storage imports made every task unusable

- Date: 2026-08-13
- Severity: P0 / allocation and execution availability
- Status: fixed on `g11-integration-reconcile`
- Evidence: a debug diagnostic using the locked environment under
  `/projects/bsm/software/rfm-pipeline` failed to finish importing
  `rfm_pipeline.hpc_campaign_package` after more than 15 minutes. A timed
  import trace spent more than 120 seconds in basic standard-library imports
  before reaching any scientific kernel. The same exact checkout and lock on
  `/scratch` installed in 26.08 seconds and completed the full campaign-module
  import in 8.96 seconds by direct Python invocation (5.35 seconds through
  warm `pixi run`).
- Resolution: campaign configuration now binds both immutable runtime
  checkouts to `/scratch/dhetting/bsm_runtime/software`; generated worker,
  audit, reducer, and retry scripts invoke the locked environment's Python
  executable directly. A regression rejects `/projects` runtime roots or
  generated `pixi run python` task launches. The scheduler diagnostic request
  is reduced from 25 minutes to 3 minutes estimated / 3 minutes 45 seconds
  requested.
- Scientific impact: none. Source, lock, Git commit, adapter, config, and data
  hashes remain preflight-bound; only runtime placement and process launch
  changed.
- Validation: full focused G11 test surface and live debug diagnostic chain
  required before scheduler readiness is restored.
- Blocks submission: until the new pinned commits are installed and the timed
  worker/audit/reducer smoke passes from the scratch runtime.

## REVIEW-0017 — Pilot AU selector and campaign envelope omitted scheduler facts

- Date: 2026-08-13
- Severity: P0 / final-run allocation correctness
- Status: fixed on `g11-integration-reconcile`
- Evidence: Kestrel `debug`, `short`, and `standard` are `OverSubscribe=EXCLUSIVE`;
  live job `16144435` requested two CPUs but `sacct` reported one node, 104
  allocated CPUs, and `billing=1024`. The pilot selector nevertheless priced
  most profiles as CPU/memory fractions. It used in-process elapsed/RSS without
  requiring matching scheduler completion, and campaign forecasts counted one
  reducer per stage while the submitted DAG runs both an audit and a reducer.
- Resolution: pilot acceptance now requires complete, successful top-level
  Slurm accounting joined to the exact worker job ID. Selection uses allocated
  nodes times scheduler elapsed and the greater of process RSS or batch-step
  MaxRSS. The retained accounting bundle hashes raw `sacct` rows, the complete
  step-to-job map, joined telemetry, worker AU, and total pilot AU. Campaign
  accounting now includes both audit and reducer jobs per stage.
- The pilot kernels now match production dimensions that affect scaling: 367
  enriched sparse/terminal candidates, five applied-model holdout/bootstrap
  passes, and the scheduler-granted CPU count. Bootstrap work units encode
  draws times outputs times models; resolution scaling counts the single nested
  maximum-draw computation rather than double-counting its B and 2B prefixes.
- Scientific impact: fixes profile choice, memory sizing, and whole-campaign AU
  admission before the one-shot final run; it does not change scientific
  estimands or methods.
- Validation: prospective exclusive-node selection, scheduler-join, RSS, and
  audit-accounting tests fail on the prior implementation and pass after repair.
- Blocks submission: yes until the corrected commits are deployed and a fresh
  immutable pilot package/preflight is generated.

## REVIEW-0018 — Interaction progress batches suppressed requested parallelism

- Date: 2026-08-13
- Severity: P0 / final-run feasibility and AU correctness
- Status: fixed on `g11-integration-reconcile`; post-fix pilot required
- Evidence: Kestrel pilot job `16148877` requested 80 CPUs for a 100-draw
  interaction block but timed out at the debug limit after averaging only about
  3.6 CPU cores. The shared score-only production kernel selected progress
  batches as `total_scores // 20`, capped at eight, so the 100-draw profile
  exposed only five concurrent tasks despite `n_jobs=80`; the 50- and 25-draw
  profiles exposed only two and one, respectively.
- Resolution: absent the explicit `RFM_PROGRESS_BATCH_SIZE` diagnostic override,
  interaction batches now expose at least `min(total_scores, n_jobs)` tasks.
  Seeds, score indices, checkpoint identities, deterministic result ordering,
  estimands, and statistical controls are unchanged. A prospective regression
  requires a 100-draw/80-worker block to dispatch batches of 80 and 21 scores.
- Validation: the regression failed with 21 five-task batches before the fix;
  it and all 16 interaction-discovery tests pass afterward. The 80 adjacent
  G11 HPC, interaction-wiring, and shard-reducer tests also pass, as do Ruff
  and `git diff --check`.
- Blocks submission: yes. Pre-fix pilot evidence is retained as rejected
  diagnostic history. The final resource freeze must come from a fresh package
  and complete post-fix pilot bound to the new source hash.

### 2026-08-13 live-pilot follow-up — interaction sizing is not a debug smoke test

The post-parallelism-fix 24-CPU/25-draw profile used its requested workers but
still hit Kestrel's one-hour debug limit. Interaction-score profile comparison
is therefore a bounded scientific sizing experiment, not a login/import smoke
test. Its three profiles now use `short` with a three-hour estimate and a
3h45 request; every other pilot stage remains on `debug`. This changes no
scientific control and raises the deterministic all-phase fixture only from
10,243 to 10,322 AUs, still well inside the fixed 25,000-AU ceiling.

## REVIEW-0019 — Completed interaction scoring crashed during artifact reporting

- Date: 2026-08-14
- Severity: P0 / pilot completion and downstream recovery availability
- Status: fixed on `g11-integration-reconcile`; replacement pilot required
- Evidence: Kestrel pilot job `16160079` completed all 25 interaction null
  draws over the full 12,720-pair family and wrote a self-verifying
  `ScoreOnlyInteractionArtifact`, then exited 1 when the pilot wrapper read
  nonexistent `artifact.checksum`. The same stale attribute access remained in
  `run_production_recovery_pipeline`, so Gate-B/recovery would have failed at
  the next persisted-interaction boundary even if the sizing job had passed.
- Root cause: `ScoreOnlyInteractionArtifact` deliberately exposes the canonical
  score-block identity as `payload_sha256`; it has never provided a generic
  `checksum` compatibility alias. Two callers were written against an assumed
  interface and the pilot wrapper lacked a direct completion-path regression.
- Resolution: both callers now use `payload_sha256` while retaining their
  existing result-field names. Prospective tests use payload-only artifacts and
  require the pilot result to remain JSON serializable and the recovery result
  to retain the persisted payload identities. No compatibility shim or
  scientific-method change was added.
- Preserved evidence: the failed artifact independently verifies with payload
  SHA-256 `770e4d9e071bb638afcc276254989528d94b83ac73cee9cc81aee54ec7c15faa`;
  the failed attempt is excluded from resource selection because Slurm reports
  `FAILED/1:0`. Its 17.058333 AUs remain included in campaign allocation
  accounting.
- Blocks submission: yes until exact commits pass the affected G11 interaction,
  recovery, artifact, reducer, and HPC package gates, are deployed to both
  pinned Kestrel runtimes, and a clean replacement pilot reaches exact
  `COMPLETED/0:0` coverage for every stage. The unrelated full repository gate
  was stopped at 53% without a reported failure at the maintainer's direction.

## REVIEW-0020 — Sparse pilot stages generated mismatched work units

- Date: 2026-08-14
- Severity: P0 / pilot completion and resource-freeze validity
- Status: fixed on `g11-integration-reconcile`; replacement pilot required
- Evidence: Kestrel `pilot_sparse_resample` job `16221146` exited `FAILED/1:0`
  with `ValueError: sparse stability work unit differs from its frozen full fit`. The stage-paced controller cancelled the two same-stage siblings and
  submitted no downstream work.
- Root cause: full-fit and resample records were paired by pilot repetition,
  but `_pilot_training_tables` seeded each synthetic pilot fixture from its
  stage-specific schedule hash. The production identity guard therefore saw
  different scaled inputs and PCA responses for every nominally paired record.
- Resolution: sparse pilot fixture generation derives one seed from the frozen
  config hash, a `pilot_sparse_fixture` domain separator, and the pilot
  repetition. Paired full/resample stages now reconstruct identical fixtures,
  while p0/p1/p2 remain independent. Other stages retain their existing
  schedule-derived seeds; production scientific paths and fail-closed sparse
  identity validation are unchanged.
- Validation: a prospective package regression failed before implementation
  because no cross-stage fixture contract existed and now requires byte-equal
  sparse inputs/PCA tables for matched records plus distinct fixtures across
  profiles. The focused G11 HPC, sparse-selection, train/freeze/predict, and
  recovery-comparator gates plus targeted Ruff/format checks pass.
- Preserved evidence: immutable failure bundle
  `/projects/bsm/g11_authorizations/g11-pilot-final-sizing-20260814d/`
  `failure_pilot_sparse_resample_16221146_20260814T202344Z` includes raw Slurm
  facts, controller state, package manifests/scripts, logs, and SHA-256 ledger.
- Blocks submission: yes until corrected pinned commits are deployed and a new
  isolated pilot completes every step exactly `COMPLETED/0:0`.

## REVIEW-0021 — Pilot recovery used inconsistent synthetic feature names

- Date: 2026-08-14
- Severity: P0 / pilot completion and recovery resource selection
- Status: fixed on `g11-integration-reconcile`; replacement pilot required
- Evidence: Kestrel `pilot_recovery` job `16238305` completed the production
  recovery pipeline, then exited `FAILED/1:0` while building its algebraic
  comparator design: feature `x3` was absent from a frame whose corresponding
  column was named `x003`. The controller cancelled jobs `16238303` and
  `16238304` and submitted no downstream work.
- Root cause: the pilot passed unnamed arrays to
  `run_production_recovery_pipeline`, which assigns `x0` through `x159`, then
  independently constructed comparator frames with `x000` through `x157` and
  explicit `binary_0`/`binary_1` columns. No test exercised this post-pipeline
  comparator boundary.
- Resolution: construct the canonical named pilot frames once and pass those
  same frames to both production recovery and algebraic materialization. A
  prospective regression requires direct, interaction, transformation, and
  binary candidates to materialize identically for training and evaluation.
  The production BSM driver was audited and already passes its exact feature
  names through both boundaries; later pilot stages do not repeat this logic.
- Preserved evidence: raw scheduler evidence and an 88-file hash ledger are at
  `/projects/bsm/g11_authorizations/g11-pilot-final-sizing-20260814e/`;
  the ledger SHA-256 is
  `49f6b2257b13b2367335ccf2bab609205b351ea974ad71299f42a88bfc413e30`.
  Rejected-attempt use is 77.91944444444445 AUs; cumulative sunk campaign use
  is 192.55 AUs.
- Blocks submission: yes until the corrected exact commits and updated
  content-addressed package pass focused validation, are deployed cleanly to
  Kestrel, and a fresh isolated pilot reaches exact `COMPLETED/0:0` coverage.

## REVIEW-0022 — Confirmatory admission double-counted completed work

- Date: 2026-08-15
- Severity: P0 / final-run allocation admission
- Status: fixed on `g11-integration-reconcile`; Kestrel replay required
- Evidence: a prospective B=999 confirmatory package stopped with
  `telemetry-based whole-campaign projection exceeds allocation_quota: 25215 > 25000`, before the exact 24,929.82-AU BSM certificate could be generated. No
  scheduler job was submitted.
- Root cause: the generic package envelope retained estimated pilot and
  resolution charges after those phases had completed. The downstream BSM
  certificate correctly replaced them with observed AUs, but the earlier gate
  made that certificate unreachable.
- Resolution: the confirmatory-only admission path accepts immutable observed
  charges for completed work and the bounded postprocessing reserve, then adds
  the shared 20% reserve only to stages present in the confirmatory package.
  Legacy package modes retain their existing admission rule. Invalid inputs
  and actual quota overruns remain fail-closed.
- Validation: prospective B=999 admission and true-overrun regressions pass;
  the complete repository gate passes, including all manuscript notebooks,
  docs, package build, hygiene, and diff checks.
- Blocks submission: yes until exact commits are pushed/deployed and fresh
  Kestrel diagnostics admit B=999, reject B=1,998, and pass `sbatch --test-only` for every unique production script without creating a scheduler
  job.

## REVIEW-0023 — Scientific isolation broke the Kestrel allocation probe

- Date: 2026-08-15
- Severity: P0 / live preflight execution
- Status: fixed on `g11-integration-reconcile`; fresh Kestrel replay required
- Evidence: the first exact same-day preflight raised
  `CalledProcessError: aus_report returned non-zero exit status 1` before phase
  authorization. Reproduction showed inherited `PYTHONNOUSERSITE=1` prevented
  the NREL utility from importing its user-site `jwt` dependency. With that one
  variable absent, `aus_report` returns normally. No scheduler job,
  authorization, or submission journal was created.
- Resolution: remove `PYTHONNOUSERSITE` only from the external `aus_report`
  subprocess environment. Scientific controller and worker Python isolation is
  unchanged, and every generated worker still clears `PYTHONPATH`/
  `PYTHONHOME` and sets `PYTHONNOUSERSITE=1`.
- Validation: a prospective regression failed before the change and now passes;
  focused live-smoke, capacity, observed-AU admission, Ruff, and diff checks
  pass.
- Blocks submission: yes until a new controller revision and new immutable
  control root complete same-day live preflight and independent evidence review.

## REVIEW-0024 — Final adapter used a nonexistent score-artifact checksum

- Date: 2026-08-15
- Severity: P0 / final campaign continuation
- Status: paired BSM fix committed; updated runtime pin under validation
- Evidence: every task in final resolution array `16278789` completed its
  persisted score block, then failed when the BSM adapter read
  `ScoreOnlyInteractionArtifact.checksum`. The generic artifact contract
  exposes `payload_sha256`; the equivalent pilot-wrapper defect was already
  fixed in REVIEW-0019.
- Resolution: paired BSM commit `67fe881` uses the canonical payload identity
  in all three affected paths and adds content-bound promotion of completed
  resolution scores into a fresh run. This RFM change updates only the reviewed
  BSM runtime path and adapter digest; the accepted scientific RFM source and
  lock remain unchanged.
- Acceptance: the direct-runtime regression requires the exact BSM checkout
  suffix and adapter SHA-256. The complete G11 package gate, clean deployment,
  independent no-submit audit, and exact live cache-promotion replay remain
  required before continuation submission.
- Blocks submission: yes until the updated pin is committed/deployed and those
  gates pass.

## Independent review of alignment diff b96442b..e31fb15 (round 3 remediation)

- REVIEW-R3-01 \[BLOCKING, fixed-by R3-S01\]: multiplicity_controlled_interaction_selection
  implemented+tested (P0-S08) but never called by production discover_manuscript_interactions
  (manuscript_stages.py:2529 still uses uncorrected per-pair quantile rule). F5 defect still shipped.
- REVIEW-R3-02 \[BLOCKING, fixed-by R3-S02\]: nonlinear_discovery_with_multiplicity_correction
  (P0-S10) not wired into production discover_manuscript_nonlinear_transformations; production
  \_score_one_nonlinear_feature uses uncorrected p-threshold + same-data transform choice.
- REVIEW-R3-03 \[MAJOR, fixed-by R3-S01\]: permutation-adequacy guard family_size=n_pairs is
  inconsistent with the per-pair retention it guards; unrunnable when ON or disabled in tests.
- REVIEW-R3-04 \[MAJOR, fixed-by R3-S03 + backlog\]: test_P0_S13 does not scan src for case-study
  literals; pre-existing AFSC/UAEORO scenario helpers remain in data.py (generalization backlog).
- REVIEW-R3-05 \[MINOR, fixed-by R3-S02\]: best_edf/best_p joint update -> best_p not min p-value.
- REVIEW-R3-06 \[MINOR, fixed-by R3-S04\]: categorical levels=None not persisted at fit -> predict
  round-trip breaks on small scenario-contrast batches (F1).
- REVIEW-R3-07 \[MINOR, fixed-by R3-S05\]: elasticnet_interactions fabricates p=0/1 + zero null
  summary instead of NaN/None.
- Verified-correct (no action): S07 adequacy math, S08 bh_fdr/fwer standalone, S04/S05/S11 sealed
  guard + leakage-free support selection, S09 provenance retained-set, M3 stratified bootstrap.
