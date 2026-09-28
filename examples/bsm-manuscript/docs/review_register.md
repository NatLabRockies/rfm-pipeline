# Review Register — BSM Public RF

Durable record of audit findings requiring follow-up. Newest first.

## 2026-08-22 — Fixed-family adapter violated detector partitions and confirmation failed

- **Severity:** BLOCKER (publication calibration gate). Generation 11 completed
  the fixed-family supplement but produced 12/200 false-selection events for
  the 100-pair family; its one-sided Wilson upper bound was 0.0939310661 above
  the frozen 0.09 gate. The threshold remains unchanged and downstream work is
  not publication evidence.
- **Root cause:** the production adapter pooled incomparable raw TreeSHAP and
  HC3-studentized binary-factorial scores into one maxT family and compared the
  result with overall alpha 0.05. The method contract instead requires a
  separate maxT calculation per detector family at its assigned family alpha.
- **Correction:** partition by persisted detector identity, apply maxT inside
  each score scale, freeze a prospective 0.020/0.020 allocation, and increase
  the precommitted fixed-family sample from 200 to 300 while retaining the
  0.09 Wilson gate. The scientific contract advances from v10/generation 11 to
  v11/generation 12, so every affected seed changes with the contract hash.
- **Evidence conservation:** the failed 200-task supplement and the prior
  5,600-task Gate B remain immutable development evidence. Diagnostics using
  those bytes support design selection only; they may not be relabeled or
  reduced as confirmation. Failed diagnostic pair-ordering attempts are also
  retained as packaging evidence and excluded from scientific decisions.
- **Acceptance criteria:** focused partition and contract-drift regressions,
  complete RFM and BSM repository gates, and immutable amendment/hash checks
  pass; clean source revisions are independently reviewed; inventories prove
  zero seed overlap; then a fresh 5,600-task Gate B and fresh 300-task
  fixed-family supplement each achieve exact scheduler/artifact coverage and
  passing reducers before downstream execution resumes.
- **Disposition:** implementation and development diagnostics are in progress
  on isolated branches. No fresh scientific phase has been submitted.

## 2026-08-18 — TreeSHAP is underpowered for the strong BB parity regime

- **Severity:** BLOCKER (canonical Gate-B power gate; all downstream analysis
  remains stopped). The complete development run recovered only 103/200 strong
  binary--binary interactions, with a one-sided 95% Wilson lower bound of
  0.4570604894 versus the frozen 0.80 gate.
- **Root cause:** after additive main-effect conditioning, the two binary
  predictors form a pure 2x2 parity contrast. Greedy tree splitting has zero
  population marginal gain at the first split, so TreeSHAP interaction scores
  are driven by noise-induced splits even though the planted interaction is
  strong and the four-cell design is balanced.
- **Correction:** assign detector identity from the training design; retain
  TreeSHAP for CC/BC pairs and use the absolute HC3-studentized saturated 2x2
  interaction coefficient for BB pairs. Apply shared-response maxT separately
  at 0.025 within each detector family, preserving an overall 0.05 bound by the
  union bound. Require all four cells and at least two rows per cell.
- **Evidence conservation:** preserve all prior results as development and
  ablation evidence, but do not reuse them as confirmation after this
  data-informed method change. Earlier data/interface/Gate-A work remains
  valid. A new Gate B uses new contract-derived seeds and no downstream work
  begins before it passes.
- **Acceptance criteria:** focused statistic, invariance, adequacy, partitioned
  maxT, artifact identity, distributed reduction, and adapter tests pass; full
  repository gates pass; clean runtime hashes and an exact AU estimate are
  audited; then a fresh 5,600-task Gate B passes all null and power rules.
- **Disposition:** prospective code/config/manuscript correction implemented in
  isolated worktrees; full validation, clean revision publication, AU review,
  and confirmation submission remain.

## 2026-08-16 — Binary-only screens failed at the nonlinear stage

- **Severity:** BLOCKER (seven Gate-B null tasks failed; downstream remained
  held). A valid empirical screen can retain only the two-point binary inputs.
  Such a support has no non-collinear supported nonlinear transformation, but
  the pinned generic stage raised instead of emitting its vacuous terminal
  result.
- **Root-cause correction:** the campaign adapter intercepts only the exact
  empty-family terminal, independently proves the supported candidate family
  is empty, validates training/PCA invariants, and returns schema-complete
  empty nonlinear artifacts. The patch is scoped to one worker call and always
  restores the pinned generic function; all other exceptions fail closed.
- **Evidence conservation:** 4,921 successful Gate-B results are retained
  without mutation or rerun. Seven failed attempts and charged cancelled work
  are diagnostic/accounting-only. A provenance-preserving continuation must
  run exactly the 679 missing tasks and combine them with the original success
  hashes under the corrected adapter/reducer identities.
- **Acceptance criteria:** binary-only and scoped-restoration regressions pass;
  the complete BSM suite passes; failed/cancelled AUs are added to campaign
  accounting; the continuation preflight validates all 4,921 cached results;
  all 5,600 Gate-B records finish with exact scheduler/artifact coverage before
  a replacement audit/reducer or fixed-family work begins.
- **Disposition:** code correction and local validation complete; deployment,
  live failed-task replay, cache promotion, and continuation submission remain.

## 2026-08-16 — Undirected interaction pair order corrupted derived recovery metrics

- **Severity:** BLOCKER (Gate-B power decision and manuscript recovery tables;
  raw scientific artifacts unaffected). In 45 of the first 87 completed
  `strong_cc` replicates, truth `HEFA:HTL` and retained `HTL:HEFA` represented
  the same undirected pair but were compared as distinct strings. Stored
  bookkeeping therefore reported 42/87 discoveries instead of the canonical
  87/87 and converted those 45 discoveries into false positives.
- **Root-cause correction:** canonicalize every interaction as an unordered
  two-factor identity; recompute Gate-B decisions from raw truth/retained IDs;
  rebuild publication recovery metrics from raw in-library truth and selected
  support rather than trusting terminal summaries. Malformed or semantically
  duplicated identifiers fail closed.
- **Evidence conservation:** old audit/reducer and dependent jobs are held,
  while worker arrays may finish because their score, prediction, model, truth,
  and selected-support artifacts are valid. No successful scientific worker is
  rerun or mutated. A continuation reducer must bind the original worker result
  hashes and the corrected reducer identity.
- **Acceptance criteria:** reversed-order unit and reducer regressions pass;
  the publication compiler corrects deliberately stale metrics; the focused
  recovery, campaign, final-execution, and publication suite passes; live
  continuation produces exact complete coverage and canonical Gate-B metrics.
- **Disposition:** code correction and local tests complete; clean commit,
  deployment, and live continuation reducer verification remain.

## 2026-08-15 — Cache validator reconstructed the wrong worker count

- **Severity:** BLOCKER (cache promotion, closed before copying or submission).
  Cached score snapshots bind `n_jobs=65`, as executed under Slurm, but the
  login-node validator reconstructed `n_jobs=1` from its own environment.
- **Root-cause correction:** cache validation now reconstructs the interaction
  contract from the manifest-bound requested CPU count and requires exact
  source/target worker-resource equality. It loads snapshot verification from
  the immutable scientific RFM checkout rather than the newer controller tree.
  Worker execution remains driven by `SLURM_CPUS_PER_TASK`; no scientific
  setting or cached byte is changed.
- **Acceptance criteria:** a deliberately mismatched login environment must
  and controller source hash must still validate 20 cache shards from their
  manifest resources and scientific runtime, report zero executed scientific
  work units, and fail if worker resources differ.
- **Disposition:** fixed in the cache-promotion path and regression-tested;
  exact Kestrel replay remains required before submission.

## 2026-08-15 — Scientific verifier rejected the plan-bound authorization schema

- **Severity:** BLOCKER (live development workers, closed before scientific
  computation). Run `g11-final-manuscript-20260815j` submitted resolution
  array `16277744`, but the first six allocated tasks stopped during
  pre-execution authorization verification. The RFM schema correctly included
  `submission_plan_sha256`; the BSM driver's exact-field set omitted it.
  Unstarted tasks and both dependent jobs were canceled immediately. Retained
  accounting records 0.036458333333333336 AU for the attempt and
  192.6780715811966 cumulative rejected-attempt AUs.
- **Root-cause correction:** BSM `d7d5f32` requires and validates the
  64-character `submission_plan_sha256`; RFM `7bc07b1` pins that exact BSM
  scientific checkout and driver digest in the Kestrel config. The regression
  test first reproduced the live rejection, then the complete BSM suite passed
  (115), the complete G11 RFM package test passed, and the exact failed
  authorization replayed successfully through the corrected scientific
  verifier on Kestrel.
- **Acceptance criteria:** a new isolated campaign must bind every manifest
  record to BSM `d7d5f32`, pass all-script `sbatch --test-only`, perform the
  same-day development preflight without submitting, and replay the exact live
  authorization through every resolution record before `sbatch` is permitted.
- **Disposition:** closed by `g11-final-manuscript-20260815k`; no-submit audit
  SHA-256 `8f64b94be45ed57651a4b149a54096b6db12aa9e5de8f4585a48722d166f3ecd`
  and authorized-no-submit replay SHA-256
  `90763ffdff43f2091520232115b89fd5f864dd8d5148fad5b22b55b78a8a2634`.

______________________________________________________________________

## 2026-08-15 — Generated workers referenced the wrong authorization path

- **Severity:** BLOCKER (live development workers, closed before scientific
  computation). Run `g11-final-manuscript-20260815h` wrote the accepted
  authorization at `development/authorization.json`, while its manifest rows
  referenced `development.json`. All 20 workers failed closed before science;
  retained accounting records 0.09161324786324784 AU for the attempt.
- **Root-cause correction:** RFM `7cc6f09` generates the canonical
  `<run>/<phase>/authorization.json` path. BSM `b3b2377` independently checks
  every target-phase manifest path before live smoke or submission and uses
  canonical Slurm `JobID` rows for array accounting.
- **Acceptance criteria:** all development and confirmatory records must bind
  their exact phase authorization target; Kestrel replay must identify every
  array task from canonical `JobID`; a new isolated run and fresh preflight are
  required.
- **Disposition:** closed by regression tests and exact Kestrel replay. Failed
  run evidence remains immutable under its control root and is excluded from
  scientific resource selection.

______________________________________________________________________

## 2026-08-15 — Isolated controller environment broke `aus_report`

- **Severity:** BLOCKER (live preflight, closed before submission). The first
  exact preflight inherited `PYTHONNOUSERSITE=1` into NREL's external
  `aus_report` utility, which then could not import its user-site `jwt`
  dependency. No authorization, submission record, journal, or scheduler job
  was created.
- **Root-cause correction:** RFM controller `77a8e80` removes
  `PYTHONNOUSERSITE` only for the `aus_report` subprocess. Controller and worker
  Python isolation is unchanged, including explicit worker removal of
  `PYTHONPATH`/`PYTHONHOME` and continued `PYTHONNOUSERSITE=1`.
- **Acceptance criteria:** a new content-addressed control root must obtain and
  hash same-day allocation output, verify the `nationalpfa` association, pass
  exact-script `sbatch --test-only`, issue phase authorization without a job,
  and remain queue/journal empty until that evidence is independently checked.
- **Disposition:** closed by run `g11-final-manuscript-20260815h`; independent
  preflight audit SHA-256
  `bdaaa415d4c82a67446828a385e1e801e32cb0502c57990a080386bdaf71c7a3`.

______________________________________________________________________

## 2026-08-15 — Final-package admission double-counted completed work

- **Severity:** BLOCKER (scientific execution, closed before submission).
  A prospective B=999 confirmatory-package build stopped at 25,215 AUs even
  though the independently computed whole-campaign maximum was 24,929.82 AUs.
  The generic package gate was reserving pilot and development estimates after
  those phases had completed, so the exact post-resolution budget certificate
  could not be reached. No scheduler job was submitted by the failed
  diagnostic.
- **Root-cause correction:** confirmatory package generation now accepts the
  observed allocation for completed work and the separately bounded
  postprocessing reserve. Its admission total is completed observed AUs plus a
  20% reserve on unexecuted confirmatory stages plus the postprocessing reserve.
  Legacy development/full-package admission is unchanged. The BSM controller
  supplies accepted-pilot, rejected-attempt, and observed development AUs from
  immutable evidence rather than estimates.
- **Acceptance criteria:** the exact deployed commits must (1) pass focused and
  repository validation, (2) generate the prospective B=999 package below
  25,000 AUs, (3) reject B=1,998 above 25,000 AUs, (4) pass `sbatch --test-only` for every unique production script, and (5) leave `squeue` and
  the submission journal empty until a fresh same-day authorization is
  independently verified.
- **Disposition:** closed by exact-commit audit SHA-256
  `95765dae4cbabbf16c3507354387397089667213b5f1221de20ef2466d457f02`;
  B=999 admitted at 24,929.819551282053 AUs, B=1,998 rejected at 49,127.8
  AUs, and all 46 confirmatory scripts passed Kestrel `sbatch --test-only`.

______________________________________________________________________

## 2026-08-11 — G11 integration reconciliation remains blocked

- **Severity:** BLOCKER (integration/review sequencing). Isolated worktree
  `/Users/dhetting/src/bsm-public-rf-g11-integration` was created at required
  G10 base `47849a518fa7a90dc29ec5bff41d86774768beac`; its existing G10 repair
  diff is preserved there without changing the source worktree.
- **Retained:** compatible G11-G0 config from `c2d6731`,
  `configs/g11_campaign_contract.toml`. It is `OPEN`, uses
  `max_stat_adjusted_p_mc`, `B_screen=3199`, `B_interaction=999`, and all
  scenarios retain 158 continuous plus two binary inputs. It has no scheduler
  command or result claim.
- **Rejected:** control commits `5e5bfcc` and `fee4368` conflict with the
  newer uncommitted G10 control snapshot/dispatch bundle. The latter must remain
  authoritative because it preserves the G10 no-results/no-PASS controls.
  `c347370` is also rejected: its HPC manifest pins displaced commit identities
  and embeds `sbatch`, so it would be stale and cannot satisfy no-submit
  readiness in this integration worktree.
- **Validation:** G11 config static assertions passed; `tests/test_bsm_recovery_fwer.py`
  passed (18). The combined targeted run had 24 passing tests and five
  failures solely for absent regenerated PDF figures. Figures/results were not
  regenerated or restored because they are quarantined, and full gates are red.
  No commit, push, scheduler submission, manuscript-result edit, or PASS claim.

______________________________________________________________________

## 2026-08-10 — G0/B pre-execution control repair

- **Severity:** BLOCKER (scientific execution). The recovery-study controls were
  rebuilt from the current handoff in a fresh worktree. G0, A, and B remain
  **OPEN**.
- **Implemented:** tracked control snapshot + checksum manifest; strict
  canonical contract; scenario-keyed development seed ledger; typed 160-field
  DGP; four binary-cell validation in both splits; bounded row-level
  heteroscedasticity; typed truth; canonical pair construction; fail-closed
  terminal-ledger validation; null aggregation that retains empty-family
  records and permits one-pair families; status-only manifests.
- **Quarantine:** unsupported recovery-study CSV/JSON/log outputs were removed
  rather than relabeled as evidence. No calibration, scheduler, HPC,
  production, holdout, or result-generation task was run.
- **Remaining blockers:** the paired generic production adapter/reducer must
  reach its pinned clean commit and receive independent G0/A review before any
  development execution or later gate is considered.

______________________________________________________________________

## 2026-09-07 — G11/G12 reconciliation (RESOLVED): main and codex chain made contradictory method claims

- **Severity:** HIGH (blocks branch reconciliation; two divergent scientific contracts in one repo).
- **Context.** The `bsm-public-rf-g11-*` / `-g12-*` directories are **git worktrees**, not separate repos — all objects already live in `bsm-public-rf/.git`. Reconciliation is branch integration, not repo consolidation. Two lines diverged at `47849a5` (2026-08-09): `main` (+6 commits → `fee4368`) and a linear 48-commit chain tipped by `codex/g12-interaction-reconcile` (`963ec4f`, 2026-08-24). `git cherry` confirms neither side contains the other.
- **Completed this session (safe, reversible):**
  - All 20 branch tips tagged `reconcile-backup/20260907-060905/*` in both repos.
  - Uncommitted base-worktree work (43 files) committed to `reconcile/base-worktree-snapshot-20260907-060905` (`d19b4d5`) — was at risk of loss.
  - `main` reconciled with `origin/main`'s 2 commits → `ca35ece` (3 conflicts resolved: kept main's B999 Gate-B settings; adopted the reproducible rfm-pipeline git pin over the local editable path).
- **BLOCKER — the chain merge cannot proceed without an author decision.** `codex/g12-interaction-reconcile` contains `test_runtime_surfaces_have_no_legacy_158_exact_or_free_text_gate_claims`, which asserts that `scripts/run_bsm_recovery_study.py`, `configs/bsm_dgp_contract.yaml`, and `configs/manuscript_case_study.yml` contain **none** of: `fwer_max_stat_exact`, `min_exact_permutation_draws`, `158-continuous`, `representative replicate`; and that `artifacts/recovery_study/` holds **only** `README.md`. The `main` line requires exactly the opposite: `StudyScale(family_error_method="fwer_max_stat_exact", min_exact_permutation_draws=199)`, `first_order_candidate_design()` over 158 continuous inputs, `family_error_method: fwer_max_stat_exact` in `manuscript_case_study.yml`, and 7 committed result CSVs under `artifacts/recovery_study/`.
- **Consequence.** Merging in either direction requires *either* deleting committed scientific results and retiring the `fwer_max_stat_exact` runtime surface, *or* weakening the chain's guard test. Both are prohibited without explicit approval, so the merge was aborted and the working tree restored clean.
- **Silent-loss hazard found.** In the trial merge, `configs/manuscript_case_study.yml` **auto-merged without conflict** and silently dropped `family_error_method: fwer_max_stat_exact`. Any future reconciliation must assert this key explicitly rather than trust auto-merge.
- **Evidence preserved, not yet applied.** Trial resolutions saved outside the repo: the retired `configs/method_contract.yaml` archived as `docs/archive/method_contract_G0_retired.yaml` (it is the **only** carrier of the P9-A-S2 independent-review PASS record — the chain's replacement configs do not contain it, so accepting the chain's deletion as-is would destroy that evidence); and a union merge of the two independently-authored `docs/execution_control/control_manifest.json` files (safety flags left locked).
- **Required action (author decision):** declare which method contract is canonical — the chain's campaign-contract architecture (`fwer_max_stat_exact` retired from runtime surfaces, recovery artifacts purged) or main's direct-runtime formulation. The reconciliation cannot be completed mechanically.
- **Disposition:** RESOLVED 2026-09-07 — see the resolution entry below.

### Resolution (2026-09-07): the G11/G12 campaign chain is canonical

- **Decision (author).** The chain's campaign-contract architecture is canonical. `main` was fast-forwarded to the reconciled tip `7eefe06`.
- **Deciding evidence.** `bsm-public-rf-manuscript/generated/manuscript_results.tex` is headed *"Generated from one immutable G11 campaign; do not hand-edit"* (run `g11-final-manuscript-20260815p-g12-20260822a`). The manuscript's `sec:recovery` numbers therefore derive from the G11/G12 campaign, **not** from main's committed `artifacts/recovery_study/` bundle. The bundle is superseded on every axis: 2 scenarios vs 12; 100 null replicates vs 1,000; `B=199` vs a 3,199-draw screening schedule; a two-sided Wilson interval vs the one-sided 95% Wilson upper bound the manuscript actually reports. `docs/ANALYSIS_HANDOFF.md:1317` independently flags the two-sided `_wilson_ci` as a mislabeling defect, which the chain fixed.
- **Runtime evidence.** Measured on the pre-merge `main`: the `interaction_null` gate replicate cost **99.7 s**, so `fwer_reps=40` made `pixi run pytest tests/` an approximately **66-minute** job (`global_null` short-circuits at 0.15 s because no signal is screened). The chain deliberately moves scientific execution to the HPC campaign so the gate stays a smoke test. Post-merge the full suite is **150 passed in ~33 s**. Note this is *not* the CPU-hours tier — that remains the campaign itself (B=3,199, ~30k runs, 23,495 outputs).
- **Resolutions applied.** `scripts/run_bsm_recovery_study.py` and `tests/test_bsm_recovery_fwer.py` taken from the chain (**no test weakened** — the guard test is intact and passing); `configs/method_contract.yaml` deletion accepted **after** archiving the P9-A-S2 review PASS record to `docs/archive/method_contract_G0_retired.yaml`; `docs/execution_control/control_manifest.json` taken from the chain (the earlier union merge broke `verify_pinned_control_snapshot`, since that manifest is hash-pinned by `control_snapshot_sha256`, and main's file is an unrelated G0-era schema); `pixi.toml` uses the chain's `rfm-pipeline` pin `fb8b57f`, which matches the API the chain's code expects.
- **Silent-loss hazard closed.** The `family_error_method` auto-merge hazard is moot under this decision: removing the key from runtime surfaces is intended, and the chain's own guard test asserts its absence, so no additional assertion is required.
- **Superseded branches confirmed, not merged.** `g11-g0-s1-contract` (an earlier, smaller draft of `g11_campaign_contract.toml`), `g11-hpc-s1-hpc-package` (a pre-submission manifest pinned to the old tip `fee4368`, with `campaign_submit = "DISABLED_UNTIL_GATE_B_PASS"`, superseded by the completed campaign), and `p9-c-s1-bsm-calibration` (its `correlated_null` regime now ships as `correlated_interaction_null` among the 12 contract scenarios).
- **R4B-S02 dropped as superseded.** It implemented the retired two-sided-Wilson, two-scenario calibration; the manuscript uses the campaign's one-sided bound. Recoverable on `reconcile/base-worktree-snapshot-20260907-060905`.
- **Worktrees.** All 10 removed; `bsm-public-rf` is now a single clean worktree on `main`. Uncommitted work was first preserved to `reconcile/g10-repair-wip-20260907`, `reconcile/g11-hpc-s1-wip-20260907`, and `reconcile/g12-publication-wip-20260907`. The manuscript repo is likewise a single worktree on `main`.
- **Follow-up (non-blocking).** Triage the three `reconcile/*-wip-20260907` branches; `configs/hpc/g11_hpc_package_config.toml` and `scripts/hpc/generate_g11_package.py` are absent from `main` and may still be wanted. All pre-reconciliation tips remain tagged `reconcile-backup/20260907-060905/*`.
- **Not pushed.** This repo is ADVISORY autonomy; all work is local pending author review.

______________________________________________________________________

## 2026-07-30 — Corrected 30k run promoted: canonical re-baseline 123 → 245

- **Severity:** HIGH (headline scientific numbers). The corrected main-effect-conditioned interaction method (rfm-pipeline 0965994) 30k Kestrel run (`publication_full_dataset_distributed_20260723`) supersedes the prior 123-feature canonical run. Artifacts, canonical config, consistency tests, and the manuscript were re-baselined.
- **Number changes (old 123-run → corrected 245-run):** final predictors 123→**245** (main 52→**63**, interaction 49→**159**, transformation 22→**23**); enriched candidate 157→**367** (70 first-order + 272 interactions + 25 transforms); stable/HC3-retained 157→**360**; delta-pruned 34→**115**; interaction pairs discovered 62→**272**; holdout NRMSE 0.0714→**0.0679** (CI [0.0699,0.0724]→**[0.0663,0.0690]**); penalized OLS 0.0706→**0.0682**. Unchanged: 23,495/9,954 outputs, 17 PCA comps, 70 screened terms, null 0.1653, main-effects/screened OLS 0.0812, seed 123.
- **NARRATIVE CHANGE requiring author review:** old text claimed "the marginal-impact pruning step is the *only* stage that removes enriched terms." In the corrected run sparse selection + stability also removes 7 terms (367→360), so §Results and §Reduction-counts prose were rewritten to state both sparse-selection (7) and delta-pruning (115) remove terms while HC3 removes none. Interactions are now the majority of the support (159/245 = 65%). Author should confirm the reframing reads correctly.
- **Tooling fix (`scripts/build_metadata.py`):** `_strip_transform` / `cross_reference_inputs` only handled prefix transform naming (`sqrt_x`); the pipeline emits suffixes (`x_sqrt/_sq/_log1p/_inv`), so 23/245 transform features were mislabeled `identity` and self-referenced their base input. Added suffix handling + focused tests (`tests/test_build_metadata_transforms.py`, 10 pass). Metadata regenerated: 0/245 unmatched.
- **Stale artifact REMOVED:** `artifacts/final_model/feature_drop_noref_nrmse_impact_full30k.csv` (506 rows, prior 132-model) was an orphaned one-off diagnostic — the corrected pipeline does not emit it (Kestrel `feature_pruning` stage produces only `feature_pruning_impact.csv`), it is not manuscript-cited, not referenced by any test or reproduce script, and is not regenerable without a bespoke full-30k-data script. Removed from the release bundle (`git rm`) and dropped from the final_model README rather than ship stale/non-reproducible values.
- **Config-echo correction:** `final_ols_summary.csv` carried pipeline-echoed `manuscript_*_reference` cells = 132/0.0721 (from a stale Kestrel case-study config, matching neither canonical). Corrected to 245/0.0679 to match the re-baselined manuscript; actual run outputs untouched.
- **Validation:** full bsm-public-rf suite 53 pass (incl. re-baselined `test_artifact_bundle_consistency.py`, `test_manuscript_config_reconciliation.py`). Figures + metadata regenerated via `pixi run reproduce-artifacts` / `build_metadata.py`.
- **Disposition:** artifacts + config + tests updated in bsm-public-rf; `manuscript.tex` updated in bsm-public-rf-manuscript. Both **local only, uncommitted** (advisory repo — awaiting owner approval to commit).

______________________________________________________________________

## 2026-06-07 — Round 21: HPC reduce entrypoint imports absent tool module

- **Severity:** HIGH (when invoked from a pip-installed rfm-pipeline). `pixi run rfm-hpc-reduce --help` failed at import time in both rfm-pipeline (source tree) and bsm-public-rf (pip install): pinned `rfm_pipeline.hpc_reduce` imports `tools.run_manuscript_pipeline`, but `tools/` is not part of the installed package.
- **Round-21 partial fix (rfm-pipeline 7eea471):** corrected `REPO_ROOT = Path(__file__).resolve().parent.parent.parent` (was `.parent.parent`, which only added `src/` to sys.path). Now `rfm-hpc-reduce --help` works from a source-tree checkout. Verified locally.
- **CLOSED (round 22, rfm-pipeline 6826edd, bsm-public-rf 18b4555):** helpers moved into `src/rfm_pipeline/manuscript_pipeline_helpers.py` (installable package). `hpc_reduce.py` now imports cleanly without sys.path hacks. `pixi run rfm-hpc-reduce --help` verified in install mode under bsm-public-rf. Smoke tests added in rfm `tests/test_docs_snippets_smoke.py` to guard the regression.
- **Further cleanup (rfm-pipeline e53f9c0):** duplicate helper bodies removed from `tools/run_manuscript_pipeline.py`; that script now imports from the package (-370 LoC, single source of truth).

______________________________________________________________________

## 2026-06-07 — Round 20: pullback bundle helper missing

- **Severity:** HIGH. `scripts/kestrel/pull_hpc_artifacts_bundle.sh` calls `tools/hpc_bundle_manifest.py` for create/analyze/metadata, but that file is absent from HEAD and `git ls-files`. `pixi run hpc-workflow ... --action collect --dry-run` still emits this broken command, so real collect/study-package pullback fails after HPC work completes. Required follow-up: restore/track the helper or replace the pullback path with existing `rfm_pipeline` bundle tooling and add a local test that referenced helper paths exist.

______________________________________________________________________

## 2026-06-07 — Round 16: HPC dry-run only validates 1 of 6 stages

- **Severity:** HIGH (silent false-confidence in reproducibility).
- **Evidence:** `prepare_full_pipeline_artifacts: true` in
  `configs/hpc/kestrel_publication_orchestration.yml` and
  `configs/hpc/dev/kestrel_workflow_small_distributed.yml` has **zero code
  consumers** in `scripts/hpc_workflow.py` (grep confirmed). Only
  `stage:` and `prepare_interaction_inputs` are honoured. The
  documented dry-run cascade through all 6 stages does not happen via
  the orchestration YAML — only `output_conditioning` is validated.
- **Mechanical fix applied (round 16):** dropped the dead key from both
  YAMLs; rewrote the misleading "Generate all 6 stages" comment to state
  the actual cascade contract.
- **CLOSED (round 22, rfm-pipeline b6ed313 + bsm orchestration update):**
  added `execution.stages` (ordered list) to `HpcExecutionConfig` and
  `build_remote_submit_commands`; orchestrator now emits one
  `rfm-hpc-submit` per (stage, tier) with per-stage output dirs.
  `configs/hpc/kestrel_publication_orchestration.yml` updated to use
  `stages:` list for all 6 stages. **Companion HIGH (round 22):**
  `scripts/hpc_workflow.py` previously short-circuited remote execution
  when `--dry-run`, so the remote `rfm-hpc-submit --dry-run` (which
  generates the SLURM scripts for inspection) never ran — fixed by
  always executing the submit SSH call; the safety comes from the
  remote command's own `--dry-run` flag (commit pending in bsm).
  Cascade + dry-run propagation guarded by new tests
  `tests/test_hpc_workflow_orchestration.py:: test_submit_commands_cascade_over_stages_list` and
  `test_submit_commands_dry_run_propagates_to_rfm_hpc_submit`.

______________________________________________________________________

## 2026-06-07 — Round 16: monitor script polling loop missing

- **Severity:** MED.
- **Evidence:** `scripts/publication-run/03_monitor_publication_run.sh`
  accepted `[interval]` arg and advertised "check every X minutes"
  semantics but had no polling loop — single status check then exit.
- **Fix applied (round 16):** added optional polling loop. No arg →
  one-shot. With arg N → poll every N seconds until Ctrl-C. Trap on
  SIGINT for clean exit.

______________________________________________________________________

## 2026-06-07 — Deferred (multi-round)

- **Round 14 #4 (MED):** `configs/manuscript_runtime.yml`
  `manuscript_section:` strings stale vs v22 §3.1-3.6 numbering.
  ~9 yaml lines, doc-only.
- **Round 14 #5 (LOW):** `scripts/publication-run/README.md` and
  `QUICKSTART.md` overload "Stage" for both wrapper scripts and
  pipeline STAGES tuple. Terminology rename, not mechanical.
- **Round 15 (LOW):** `README.md` Zenodo DOI placeholder — waits on
  submission.

______________________________________________________________________

## 2026-06-08 — Round 23: cascade shard-dir collision + dependency gap

- **Severity:** HIGH (both rfm-pipeline and bsm-public-rf).
- **Evidence:**
  - rfm `hpc_submit.py:163` set `output_root = artifact_dir/hpc_shards`
    regardless of `--stage`, so cascade stages shared `_SUCCESS.json`
    markers — stage 2's `_find_incomplete_task_ids` would treat all
    shards as already complete and skip real work.
  - rfm `build_remote_submit_commands` emitted N pixi commands per
    cascade run with no SLURM `--dependency=afterok` chain, allowing
    stage N+1 to start before stage N reduce completes (race on
    canonical stage artifact).
  - bsm orchestration consumed the flat command list with no way to
    capture or thread reduce job IDs between stages.
  - bsm pullback used `bsm_*.out` log glob (job names are `rfm_*`),
    `empirical_null_screen` (HPC name is `empirical_null_screening`),
    and `cpu_nodes_<n>/hpc_scripts/manifest.jsonl` (cascade uses
    `cpu_nodes_<n>_<stage>/hpc_scripts/manifest.jsonl`).
  - bsm 04-collect `find -name "$stage"` matched nested non-canonical
    directories.
  - bsm `--dry-run` help still said "without executing" after r22 made
    it always SSH-execute.
- **CLOSED (round 23, rfm-pipeline 8ca839d + 7c33036, bsm ef49f17):**
  - rfm: per-stage shard output dirs (`hpc_shards_<stage>`);
    `rfm-hpc-submit --depends-on-job-id` flag injects
    `#SBATCH --dependency=afterok:<id>` into array and GPU array
    scripts; emits `RFM_HPC_SUBMIT_REDUCE_JOB_ID=<id>` marker on
    stdout; new `build_remote_submit_command_groups` returns
    `[(stage|None, [cmds])]` so orchestrators can chain; status
    command emits per-stage `hpc_shards_<stage>` entries; validation
    picks stage XOR stages; prep block keys off `effective_stages()`.
  - bsm: orchestrator uses the grouped builder, captures each cascade
    stage's reduce job id from the marker line, and rewrites the next
    stage's commands to append `--depends-on-job-id <id>`. Pullback
    log glob → `rfm_*`, stage name fixed to `empirical_null_screening`,
    per-stage script-dir pullback added. 04-collect tightened to
    depth-2 stage path check. `--dry-run` help rewritten with SSH
    prereq.
  - Doc drift: `docs/HPC_DISTRIBUTED_EXECUTION.md` `rfm-hpc-shard`
    typo → `rfm-hpc-worker`; rfm `README.md:94` stage names aligned
    with `_VALID_STAGES`; `paper/paper.md:107` "shipped pre-fitted
    Random Forest" rephrased to "regenerated from collected runs via
    `plot_sensitivity_rf_figures.py`".
- **Tests:** rfm 462 pass / 11 skip (added per-stage shard test +
  array-dependency injection tests + grouped-builder test); bsm 20
  pass; orchestrator smoke-import + marker parser verified.

______________________________________________________________________

## 2026-06-07 — Round 24: orchestrator parity, marker robustness, name drift

- **Severity:** HIGH (both repos).
- **Evidence:**
  - rfm `tools/run_hpc_workflow.py` never adopted the round-23
    grouped builder; the rfm orchestrator submitted cascade stages
    via the flat `build_remote_submit_commands` list with no
    dependency chaining, so jobs raced ahead of upstream reduce.
  - rfm `tools/hpc_bundle_manifest.py:463` scanned legacy
    `hpc_shards` only; missed per-stage `hpc_shards_<stage>` outputs
    after the round-23 layout change.
  - rfm `src/rfm_pipeline/hpc_submit.py:343` still built the manifest
    `output_path` against the legacy un-suffixed `hpc_shards`
    directory while workers now write to `hpc_shards_<stage>`.
  - rfm `_make_submit_all_script` reduce-only path referenced
    `ARRAY_JOB_ID` even when no array job was generated (would be
    unbound).
  - rfm `pixi.toml` task `hpc-small-test` and
    `docs/RUNNING_MANUSCRIPT_REPRODUCTION.md:57` pointed at
    `scripts/kestrel/run_small_distributed_test_local.sh` which does
    not exist in the repository (the entire `scripts/kestrel/`
    directory is absent on the rfm side).
  - bsm `scripts/hpc_workflow.py:368` silently ignored a missing /
    malformed `RFM_HPC_SUBMIT_REDUCE_JOB_ID` marker, leaving
    `prev_reduce_job_id` stale; the next cascade stage would then
    chain onto an earlier stage's reduce and race against the
    current stage.
  - bsm `_parse_reduce_job_id` returned the FIRST marker, not the
    last; sparse re-submits print the marker more than once and
    only the last reflects the actually-submitted reduce.
  - bsm pullback + 04-collect used HPC stage name
    `empirical_null_screening` for both shard dirs AND artifact
    dirs, but the canonical post-reduce artifact dir written by
    `manuscript_pipeline_helpers` is `empirical_null_screen`
    (no `-ing`). Per-stage pullback therefore copied a nonexistent
    `empirical_null_screening` directory and the 04-collect
    completeness check would never find it in correctly-collected
    runs.
  - bsm `--dry-run --generate-only` still SSHed despite docs saying
    fully local.
  - bsm tests had zero coverage for the cascade chaining helpers.
- **CLOSED (round 24, rfm-pipeline c9dbc7e, bsm pending commit):**
  - New shared module `rfm_pipeline/hpc_cascade.py` exposes
    `parse_reduce_job_id` (returns LAST marker, skips malformed),
    `inject_dependency_flag` (idempotent), and
    `CascadeChainError`. Both orchestrators import them so behavior
    cannot drift.
  - rfm `tools/run_hpc_workflow.py` now mirrors the bsm cascade
    pattern (grouped builder + capture + chain) and raises
    `CascadeChainError` on missing marker.
  - rfm `hpc_submit.py` `_build_fresh_manifest` writes the manifest
    against `hpc_shards_<stage>` to match the per-stage layout.
  - rfm `_make_submit_all_script` reduce-only path submits the
    reduce directly with no stale `ARRAY_JOB_ID` reference; log
    glob `bsm_*` → `rfm_*`.
  - rfm `tools/hpc_bundle_manifest.py` scans
    `hpc_shards_interaction_discovery` (cascade) with legacy
    `hpc_shards` fallback; log glob accepts both `rfm_*` and
    `bsm_*`.
  - rfm `pixi.toml` `hpc-small-test` task removed; doc snippet
    replaced with an equivalent `hpc-workflow --generate-only --dry-run` invocation.
  - bsm orchestrator: deleted local parser; imports shared helpers;
    raises `CascadeChainError` on missing marker; honors
    `--dry-run --generate-only` by skipping SSH (`>>> [skip ssh: ...]` log line so the operator can still review the commands).
  - bsm pullback + 04-collect: two parallel lists distinguish HPC
    stage names (for shard dirs / SLURM logs) from canonical
    artifact dir names (for `${run_dir}/${artifact_dir}` copy and
    completeness check).
  - bsm `--dry-run` help text updated to document the local-only
    `--dry-run --generate-only` combination.
  - New `bsm tests/test_cascade_chaining.py` (6 cases) covers
    helper-import wiring, last-marker semantics, idempotent
    injection, and `CascadeChainError` semantics.
- **Tests:** rfm 473 pass / 11 skipped (added test_hpc_cascade.py);
  bsm 26 pass (added test_cascade_chaining.py).

______________________________________________________________________

## 2026-06-07 — Round 25: collect exit code, pullback hardening, provenance link

- **Severity:** MED.
- **Evidence:**
  - `scripts/publication-run/04_collect_publication_artifacts.sh`
    printed "⚠️ Some artifacts missing" and exited 0, so the
    master orchestrator's downstream "WORKFLOW COMPLETE" banner
    would fire on incomplete collection.
  - `docs/manuscript_feature_catalog_provenance.md:73` referenced
    `docs/manuscripts/manuscript_impact_log.md` without making
    clear that the file lives in the upstream `rfm-pipeline` repo
    (this repo has no `docs/manuscripts/` directory). A new reader
    auditing the catalog could not locate the impact log.
  - `scripts/kestrel/pull_hpc_artifacts_bundle.sh:430` parallel
    arrays `hpc_stages[]` and `artifact_dirs[]` had no length
    guard; a single-array edit could silently mis-map HPC stage
    names to writer dir names.
  - `scripts/kestrel/pull_hpc_artifacts_bundle.sh:478` duplicated
    the cascade stage list for the per-stage log copy loop,
    reintroducing drift risk.
- **CLOSED (round 25, bsm-public-rf pending commit):**
  - `04_collect_publication_artifacts.sh` now `exit 1`s after the
    missing-artifacts summary so the master orchestrator (set
    -euo pipefail) halts before the success banner.
  - `manuscript_feature_catalog_provenance.md` makes the upstream
    repo path explicit and links to the GitHub URL.
  - `pull_hpc_artifacts_bundle.sh` asserts
    `${#hpc_stages[@]} == ${#artifact_dirs[@]}` and exits 2 with a
    clear maintainer message on mismatch.
  - Log-copy loop iterates `${hpc_stages[@]}` so the canonical
    stage list is defined exactly once in `_per_stage_files_copy`.
- **Tests:** 26 pass; bash syntax checks pass for both modified
  scripts.

______________________________________________________________________

## 2026-06-07 — Round 26: multi-tier cascade chain + summarizer port + SSH safety

- **Severity:** HIGH (chaining bug, mirrors rfm r26).
- **Evidence:**
  - `scripts/hpc_workflow.py` chained the next cascade stage to only
    the LAST tier's reduce job id (same bug class as rfm r26#1).
  - `tools/hpc_bundle_manifest.py` was the pre-r25-followup version;
    cascade runs got `scripts_missing` rows because the summarizer
    only knew about `hpc_shards_interaction_discovery`.
  - `scripts/publication-run/04_collect_publication_artifacts.sh`:
    when `$EXTRACT_DIR/runs` was missing entirely (manifest-only
    bundle), all per-stage checks were skipped and the script
    reported "All critical artifacts present" purely on the
    strength of the manifest files.
  - `tools/hpc_bundle_manifest.py` `_collect_stage_metrics` only
    looked at `runs/<target>/run_artifacts/<stage>/`. The
    `reporting_bundle` pullback mode copies stages directly under
    `runs/<target>/<stage>/`, so stage-metric columns were dropped
    from bundles produced via that mode.
  - `scripts/kestrel/status_publication_full_dataset_distributed.sh`
    interpolated `${STUDY_ROOT}` directly into the remote SSH
    command string; spaces or shell metacharacters in
    `--study-root` could break the script or inject commands.
  - `scripts/kestrel/pull_hpc_artifacts_bundle.sh` array-length
    guard accepted equal-but-empty arrays (a maintainer edit that
    cleared both lists silently copied zero stage dirs/logs).
  - `scripts/kestrel/controller_publication_full_dataset_distributed.sh`
    had ten parallel per-stage resource arrays (`N_SHARDS`,
    `WALLTIME`, `MAX_CONCURRENT`, etc.) and no length guard —
    drift would silently misassign resources to wrong stages.
- **CLOSED (round 26, bsm-public-rf pending commit):**
  - Bumped rfm-pipeline pin → 25ef483 to pick up
    `parse_all_reduce_job_ids`, multi-id `inject_dependency_flag`,
    and the cascade-aware summarizer helpers.
  - `scripts/hpc_workflow.py` now collects one id per per-tier
    invocation and threads the FULL list into the next stage as a
    colon list. `CascadeChainError` fires on partial markers too.
  - `tools/hpc_bundle_manifest.py` replaced wholesale with the
    rfm-pipeline r26 version: `_discover_stages_in_run_dir`,
    `_stages_to_summarize`, `_stage_suffixed_manifest`,
    cascade-aware `_summarize_target` with `stage=` parameter,
    cascade-aware `_collect_stage_metrics` (run_artifacts/ nested
    OR flat reporting_bundle layout).
  - `04_collect_publication_artifacts.sh` now flips
    `ALL_PRESENT=false` AND prints an explicit "MISSING: runs/"
    line when no per-target stage dirs exist (so exit 1 from r25
    actually fires for manifest-only bundles).
  - `pull_hpc_artifacts_bundle.sh` array guard now also rejects
    empty arrays (length-> 0 check before length-equality check).
  - `status_publication_full_dataset_distributed.sh` rewritten to
    pass STUDY_ROOT / STUDY_ID as POSITIONAL args to `bash -s`
    via a quoted heredoc — no more inline interpolation, no more
    triple-backslash escaping; metacharacters in --study-root are
    inert.
  - `controller_publication_full_dataset_distributed.sh` asserts
    every per-stage resource array length matches STAGES length
    via a `declare -n` loop (bash 4.3+, Kestrel default).
- **Tests:** 26 pass; bash syntax checks pass on all four
  modified scripts.

## REVIEW-0013 — G11 BSM integration and legacy-route retirement

- Date: 2026-08-12
- Severity: P0 / scientific-execution integrity
- Status: fixed in the current uncommitted integration branch
- Evidence: the prior public entrypoints could still select superseded
  screening/interaction/bootstrap controls; the full 160-input, sealed-holdout,
  terminal-support recovery, fixed-family calibration, comparator, and applied
  bootstrap route was not one executable manifest-bound adapter.
- Resolution: added the content-hashed G11 adapter and DGP contract, exact
  campaign contract, deterministic sealed applied-data preparation, terminal
  recovery/comparator flow, Gate-B/Gate-C reducers, applied stage handlers, and
  fail-closed guards on the legacy publication submission routes. The old
  method contract is audit history under `configs/rejected_history/`, not a
  compatibility fallback.
- Validation: `75 passed`; Ruff format/check passes on G11 implementation
  surfaces; guarded shell scripts pass `bash -n`; `git diff --check` passes.
- Remaining blockers: the integration diff must be committed and installed at
  exact clean Kestrel SHAs; the real applied dataset must be prepared and its
  manifest hash frozen; live quota/storage/inode and `sbatch --test-only`
  evidence are unavailable locally. No scheduler submission is authorized.
- Blocks merge/submission: merge no after review/commit; HPC submission yes
  until every remaining blocker is satisfied.

## REVIEW-0014 — Final resolution workers lost completed score work at terminalization

- Date: 2026-08-15
- Severity: P0 / final scientific campaign and allocation conservation
- Status: fixed locally; clean deployment and continuation replay required
- Evidence: all 20 tasks in final-campaign array `16278789` completed their
  score-only interaction artifacts, then exited `FAILED/1:0` at
  `g11_campaign_adapter.py:1370` because the wrapper read nonexistent
  `ScoreOnlyInteractionArtifact.checksum`. The canonical property has always
  been `payload_sha256`. The same stale access existed in fixed-family and
  applied-interaction paths. The failed attempt consumed exactly
  118.40357905982906 AUs; cumulative rejected-attempt usage is
  311.08165064102566 AUs.
- Root cause: the BSM adapter duplicated score-artifact terminalization in
  three call sites and had no payload-only regression, despite the equivalent
  RFM pilot-wrapper defect having already been documented and repaired.
- Resolution: all score paths now use one validated payload-identity helper;
  resolution terminalization is factored from score generation. A fresh
  controller may promote only the preserved, hashed resolution score blocks
  after live preflight and phase authorization. Promotion requires exact
  scientific-identity equality, verifies every artifact against the fresh
  canonical contract, writes cache provenance and atomic success wrappers,
  and leaves failed scheduler telemetry diagnostic-only. Normal workers then
  run the existing complete-stage resume validator instead of recomputing.
- Acceptance: payload-only tests cover all adapter source paths; a 20-shard
  promotion regression proves zero executed scientific work units and exact
  resume coverage; the complete BSM suite passes under both its local
  environment and the RFM scientific environment. Live acceptance additionally
  requires a new content-addressed BSM runtime/config pin, an independent
  no-submit audit, exact promoted-cache coverage, `COMPLETED/0:0` for the fresh
  no-op workers/audit/reducer, and a valid resolution decision.
- Blocks submission: yes until the clean revisions are deployed and the fresh
  continuation passes those live gates. The failed run and its job telemetry
  remain ineligible scientific/resource-selection evidence.

## REVIEW-0015 — Downstream package and environment were not executable

- Date: 2026-08-19
- Severity: P0 / post-Gate-B execution availability and provenance
- Status: fixed and locally validated; clean Kestrel deployment/dry-run pending
- Evidence: the stale package included 1,000 fixed-family tasks and projected 48,414.674145 AUs,
  while schema-v2 phase authorizations omitted the exact submission-plan hash required by the
  worker. The paired BSM environment also pinned RFM `58f3066`, which lacked the repaired
  downstream authorization contract used by the active corrected Gate B.
- Resolution: pin RFM `a3bcbaa389bc7a708ca644f4e95f9b9268407e21`, regenerate `pixi.lock`,
  rebind the rejected-history development contract and deterministic seed ledger under the
  repository's established pin-update rule, add an independently hashed Gate-B adoption bridge,
  and authorize the 200-task supplement as its own `fixed_family` phase. The RFM generator now
  creates a downstream-only DAG with no Gate-B workers and rejects authorization/plan drift before
  any scheduler call.
- Validation: exact locked install, installed-package import smoke, scoped Ruff format/lint, and
  all 126 BSM tests pass. No scientific or downstream job was submitted by this repair.

## REVIEW-0016 — Generation-12 packaging needed explicit prerequisite adoption

- Date: 2026-08-22
- Severity: P0 / confirmatory provenance
- Status: fixed in `codex/g11-fixed-family-partition`; live immutable package pending
- Evidence: the accepted pilot resource freeze, B=999 resolution decision, and
  development completion bind the superseded source and contract identities.
  Relabeling those JSON records would make the new package executable but would
  erase which evidence was reused across the generation boundary.
- Resolution: `adopt-generation-12-prerequisites` verifies every source
  self-hash and identity, preserves the exact telemetry, resource selections,
  resolution decision, and observed development AUs, then writes new
  content-addressed prerequisite records. The adoption explicitly excludes all
  prior Gate-B and fixed-family results and requires fresh zero-overlap 5,600 +
  300 task confirmation. No resource decision or scientific resolution result
  is recomputed.
- Validation: positive and stale-source negative tests pass; package generation
  must additionally verify the adopted freeze with the RFM native validator and
  prove fresh seed disjointness before execution.
