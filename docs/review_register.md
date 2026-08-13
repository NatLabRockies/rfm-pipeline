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
post-pilot test fixture projects 23,021 AUs including the completed pilot,
resolution, confirmatory production, and shared reserve.

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
