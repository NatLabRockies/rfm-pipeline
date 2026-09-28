# G11 final campaign and manuscript-release runbook

> **Generation-12 amendment (2026-08-22):** The generation-11 fixed-family
> supplement failed the unchanged 0.09 Wilson gate. It and the preceding Gate B
> are now development-only for the affected interaction calibration. The next
> confirmation freezes separate TreeSHAP and binary-factorial maxT families at
> 0.020/0.020 and exactly 300 fixed-family replicates. It requires fresh
> contract-derived seeds with zero overlap, a fresh 5,600-task Gate B, and a
> fresh 300-task fixed-family supplement. No later step in this runbook is
> permitted until both generation-12 reducers pass.

## Outcome

One restartable controller carries the accepted pilot through development
resolution, the three confirmatory gates, and a bounded artifact-compilation
job. A separate local command regenerates figures, independently audits the
bundle, installs the result surfaces into the manuscript, and compiles the two
JDS PDFs. No scientific or artifact result is hand-transcribed.

The controller is fail-closed. It performs at most one transition per
`advance` call, requires an explicit `--execute` flag for every submission,
refuses to overwrite evidence, and never advances past a failed scheduler job,
failed reducer, failed calibration gate, failed applied-data gate, failed
recovery gate, or campaign projection above 25,000 AUs.

### Verified continuation after completed scientific work

If a worker finishes an expensive, self-verifying score artifact but fails
while writing its terminal JSON wrapper, preserve the failed run before making
any repair. A fresh campaign may receive that preserved evidence through
`initialize --resolution-cache-evidence-root`. The evidence root must contain
the verified `SHA256SUMS`/`SHA256SUMS.sha256` pair, exact failed-run manifests,
source snapshots, scheduler accounting, and copied score artifacts.

The controller keeps the fresh development outputs empty through live
preflight. After phase authorization and before submission, it then promotes
the cache only when all 20 work units match on fixture, schedule index, seed,
draw schedule, family order, RFM source, dependency lock, campaign contract,
DGP, and recovery-driver identities. Every copied `ScoreOnlyInteractionArtifact`
is loaded and verified against the fresh canonical execution contract. The
controller reconstructs terminal records and atomic worker success wrappers,
then submits the normal worker array. Those workers perform only complete-stage
resume validation and exit; they do not repeat permutation scoring. The normal
audit and reducer remain mandatory.

Failed-job resource telemetry is never reused for profile selection or runtime
forecasting. Only the completed scientific score bytes are reused, with a
per-shard cache-provenance record. Later stages similarly retain valid atomic
worker results across audit or reducer failures; recovery work must resubmit
only inexpensive validation/terminalization around those retained artifacts,
not their scientific kernels.

## Fixed execution order

1. Close the completed 63-step accepted pilot from exact retained job IDs and
   raw `sacct` rows.
2. Freeze resources and account for the accepted pilot plus every rejected
   attempt.
3. Generate, preflight, submit, and verify the development-only interaction
   resolution phase.
4. Accept only a prespecified resolution decision and build a fresh
   confirmatory package whose seeds are bound to those decision bytes. Before
   package generation, apply the hashed publication amendment that changes
   only the fixed-family supplement from 1,000 to exactly 300 precommitted
   replicates; all five primary null regimes remain at 1,000 each.
5. Certify the complete package envelope, prior rejected-attempt AUs, and a
   5-AU publication-job reserve against the 25,000-AU ceiling.
6. Run Gate B calibration and fixed-family checks. A non-PASS decision stops
   the campaign before the applied analysis.
7. Run Gate P applied stages. The 1,500-row holdout remains excluded from all
   discovery, selection, pruning, fitting, and calibration.
8. Run Gate C recovery/stress simulations only after Gate B and Gate P have
   exact completion evidence.
9. Run the publication compiler as one `shared`, one-CPU, 8-GB, 30-minute
   Slurm job (maximum reserved charge: 5 AUs). This creates all tables, CSVs,
   coefficient products, figure inputs, generated LaTeX, and provenance while
   the immutable Kestrel paths are still available.
10. Transfer the compact publication directory to the laptop. Regenerate all
    11 SVG figures and the five PDFs used by the article, independently audit
    hashes and scientific identities, install results, and compile
    `manuscript.pdf` and `coverpage.pdf` locally. This final step consumes no
    allocation.

The three confirmatory phases are deliberately distinct. Gate P cannot begin
unless both Gate-B reducers report PASS, and Gate C cannot begin unless those
decisions and the applied-bootstrap summary are present and identity-bound.

## Fixed 25,000-AU branch envelope

The retained no-submit projection from accepted pilot telemetry is conditional
on the development resolution result. Completed non-array work is charged from
its exact top-level `sacct` row; array work is charged from every expected
array-task row, with exact task coverage required even when Slurm omits the
synthetic array-parent row. The shared 20% reserve is applied only to
unexecuted confirmatory work.

- At selected interaction schedule B=999, the remaining confirmatory estimate
  is 20,250.751923076525 AUs and its shared-reserve request is 24,301 AUs.
  After the accepted pilot (90.36944444444445 AUs), rejected attempts (192.55
  AUs), and publication reserve (5 AUs), no more than
  411.0805555555562 actual AUs may be consumed by development resolution.
- At selected interaction schedule B=1,998, the remaining confirmatory estimate
  is 40,415.19636752113 AUs and its shared-reserve request is 48,499 AUs. That
  branch cannot fit the allocation even before adding resolution AUs.

The 20 resolution workers retain the pilot-sized 65 CPUs, 6 GiB, and 2:43:33
walltime but use Kestrel's 104-core `shared` nodes. Including the shared
audit/reducer requests, their complete scheduler-walltime charge is at most
340.9001068376069 AUs. The corresponding whole-campaign maximum is
24,929.81955128205 AUs, leaving 70.18044871795064 AUs of hard headroom even if
every resolution job consumes its full request. The budget certificate later
replaces this maximum with the exact `sacct` charge.

The development guard checks the generated resolution package against the
reviewed B=999 confirmatory reserve. After resolution, the controller derives
the exact selected-branch quantities from the generated confirmatory package
and writes a budget certificate only if the selected branch and actual
development charge fit. Otherwise it stops before every confirmatory
submission. Completion of the article therefore remains conditional on the
prespecified resolution selecting B=999; the scientific resolution rule is not
weakened to force the affordable branch.

Before each confirmatory phase, a separate immutable `budget_guard.json`
replaces earlier-phase estimates with their newly observed AUs and recomputes
the reserve for work still unexecuted. It also requires the current phase's
full scheduler-requested walltime charge, plus actual spending and the
publication reserve, to fit below 25,000 AUs. Because the controller never
retries a failed scientific job automatically, this is a hard phase boundary:
even a phase consuming its complete requested walltime cannot cross the cap.

## One-time Kestrel initialization

Run only after the pilot submission record is `COMPLETE` and every pilot step
is exactly `COMPLETED/0:0`. Use a new run ID and a new, empty control root.
`initialize` is NO-SUBMIT and replaces the tracked placeholder run ID in a
private copy of the reviewed config.

```bash
BSM_RUNTIME=/scratch/dhetting/bsm_runtime/software/bsm-public-rf-f9493c7
RFM_SCIENTIFIC=/scratch/dhetting/bsm_runtime/software/rfm-pipeline
RFM_CONTROLLER=/scratch/dhetting/bsm_runtime/software/rfm-controller-77a8e80
PILOT_ID=g11-pilot-final-sizing-20260814f
FINAL_ID=g11-final-manuscript-YYYYMMDDa
CONTROL=/projects/bsm/g11_authorizations/${FINAL_ID}
# nationalpfa is a valid Slurm association but is not listed by aus_report for
# this non-lead user. The live-smoke contract therefore requires the configured
# allocation ceiling; rejected/accepted pilot AUs are subtracted separately by
# the immutable whole-campaign budget certificate and rolling guards.
REMAINING_AU=25000

cd "${BSM_RUNTIME}"
"${RFM_SCIENTIFIC}/.pixi/envs/default/bin/python" \
  scripts/g11_final_execution.py initialize \
  --campaign-root "${CONTROL}" \
  --run-id "${FINAL_ID}" \
  --base-config "${RFM_SCIENTIFIC}/configs/hpc/g11_kestrel_campaign.yml" \
  --pilot-package-root "/kfs3/scratch/dhetting/bsm_runs/${PILOT_ID}-package" \
  --pilot-submission-record "/projects/bsm/g11_authorizations/${PILOT_ID}/submission_job_ids.json" \
  --pilot-resource-freeze "/projects/bsm/g11_authorizations/${PILOT_ID}/resource_freeze.json" \
  --rfm-repo-root "${RFM_SCIENTIFIC}" \
  --rfm-controller-root "${RFM_CONTROLLER}" \
  --bsm-runtime-root "${BSM_RUNTIME}" \
  --prior-sunk-au 192.55 \
  --remaining-au "${REMAINING_AU}" \
  --allocation-quota-au 25000 \
  --postprocessing-reserved-au 5
```

For the one authorized continuation of
`g11-final-manuscript-20260815k`, also supply the preserved failure bundle and
the exact cumulative rejected-attempt charge:

```bash
  --resolution-cache-evidence-root \
    /projects/bsm/g11_failure_evidence/g11-final-manuscript-20260815k_scientific_adapter_failure_20260815T182652 \
  --prior-sunk-au 311.08165064102566
```

That continuation is the only campaign allowed to use this cache. Its control
manifest binds the evidence inventory before any package or scheduler action.

`RFM_SCIENTIFIC` must be a clean detached checkout at
`fd12fd579d8743bdc4acd00e1dac217cbfc56e84`, the source identity measured by
the accepted pilot. `RFM_CONTROLLER` must be a separate clean detached
checkout at `77a8e809ecb9162019c560619be3731206522c72`. Its changes from
`fd12fd5` are limited to reviewed campaign-control behavior: the
pilot-accounting schedule-hash repair, resolution-only development packaging,
package-local Slurm logs, clean compute-node imports, and exact
script/plan/accounting bindings, and partial-node placement for the
resolution-only development stage. Confirmatory admission additionally uses
observed allocation for completed pilot and development work, reserves 20%
only on unexecuted confirmatory stages, and counts the separate 5-AU
postprocessing reserve. The
controller prepends that code to its own in-process import path; it never
exports the controller checkout through `PYTHONPATH`. Real `sbatch` calls
strip Python import overrides, and generated workers continue to execute from
`RFM_SCIENTIFIC`. The control manifest records all three Git revisions, hashes
the complete RFM controller and BSM script trees, and refuses any later byte
or revision change.

The controller itself may be launched with `PYTHONNOUSERSITE=1`. Live smoke
removes that variable only from NREL's external `aus_report` subprocess because
the site utility loads its own `jwt` dependency from the login user's site
packages. Scientific imports and all Slurm workers remain isolated.

The immutable base config retains the clean scientific BSM checkout at
`1762c6bad2342dfa7e5fff632623449979963282`; same-day preflight records that
revision and verifies every referenced scientific file. The separate
`BSM_RUNTIME` checkout is controller/publication code at
`f9493c76310ea9556a4ab1bca6051802b25756d4`. The scientific adapter, recovery
driver, data preparer, DGP contract, and applied config have byte-identical
SHA-256 values in both checkouts.

Before adding `--execute`, run transitions without it and inspect the created
`control_manifest.json`, private `final_campaign.yml`, resource freeze, and
development package. The command stops before the same-day live preflight with
`EXECUTION_FLAG_REQUIRED`; this prevents an authorization from becoming stale
while awaiting manual inspection. Inspect the confirmatory package and budget
certificate when the controller creates them after the resolution phase.

```bash
"${RFM_SCIENTIFIC}/.pixi/envs/default/bin/python" \
  scripts/g11_final_execution.py advance \
  --control-manifest "${CONTROL}/control_manifest.json"
```

## Autonomous final execution

After the NO-SUBMIT inspection passes, the lightweight login-node controller
may be run at a five-minute cadence. It only invokes filesystem validation,
`squeue`, `sacct`, and `sbatch`; scientific work runs on compute nodes.
Run `aus_report` immediately before starting or restarting the controller. If
it still omits `nationalpfa`, retain the required configured ceiling value
`25000`; if it begins reporting that account, use its exact current integer.
The invocation-level value overrides the initialization snapshot for every
same-day live preflight. A stale date or a visible allocation value that does
not match the supplied integer stops before submission; refresh the value and
restart the same controller rather than creating or resubmitting a campaign.

```bash
nohup "${RFM_SCIENTIFIC}/.pixi/envs/default/bin/python" \
  "${BSM_RUNTIME}/scripts/g11_final_execution.py" watch \
  --control-manifest "${CONTROL}/control_manifest.json" \
  --remaining-au "${REMAINING_AU}" \
  --poll-seconds 300 \
  --execute \
  >"${CONTROL}/final_controller.out" \
  2>"${CONTROL}/final_controller.err" &
```

The terminal success token is
`HPC_AND_PUBLICATION_BUNDLE_COMPLETE`. Anything else is not authorization to
edit manuscript results. A terminal Slurm failure raises immediately, leaves
downstream phases locked, and preserves the submitted job record. The
controller never retries a failed scientific job automatically.

## Required Kestrel acceptance evidence

The campaign is complete only when all of the following exist and validate:

- immutable `control_manifest.json` and final config hashes;
- accepted-pilot raw accounting and allocation-bound `resource_freeze.json`;
- development submission, raw `sacct`, completion, and accepted resolution
  decision;
- confirmatory package, hashed `contract/fixed_family_amendment.json`, and
  `budget_certificate.json` with status `WITHIN_ALLOCATION`; the certificate
  must bind the amendment, replace completed pilot/resolution estimates with
  exact observed AUs, reserve 20% only on unexecuted confirmatory work, and
  count prior rejected attempts plus the 5-AU publication reserve;
- Gate B, Gate P, and Gate C submission records, exact `COMPLETED/0:0`
  coverage for every non-array job and every expected array task, reducer
  hashes, phase-specific budget guards, and phase completion records;
- Gate-B and fixed-family PASS decisions;
- applied-bootstrap completed summary and Gate-C
  `READY_FOR_INDEPENDENT_REVIEW` ledger;
- publication-job submission record and retained raw `sacct` evidence; and
- `publication_artifact_manifest.json` with status
  `PUBLICATION_ARTIFACTS_COMPLETE` and a matching file inventory.

## Publication bundle contents

The compiler emits a compact directory under
`/scratch/dhetting/bsm_runs/<FINAL_ID>/publication_artifacts`:

- `tables/`: output eligibility ledger for all 23,495 outputs, per-output
  nRMSE, summaries, ablations, model performance, workflow retention, pruning,
  empirical FWER, recovery replicates, and recovery scenario summaries;
- `final_model/`: raw-scale and standardized-X coefficient matrices,
  intercepts, standardization values, frozen support, prefilter and HC3/pruning
  ledgers, frozen-model arrays, and manifests;
- `metadata/`: the content-bound 160-input feature catalog in Parquet and CSV,
  including explicit identification of the 158 continuous and two binary
  inputs; the 23,495 output identities and exclusion reasons are in the output
  eligibility ledger under `tables/`;
- `figure_data/`: exact CSV inputs for all 11 deterministic figures;
- `manuscript/`: the one generated result-macro surface plus FWER and recovery
  table rows;
- `provenance/`: campaign contract, fixed-family amendment, path-sanitized package/inventory identity,
  path-sanitized applied dataset manifest, applied config, and scientific
  adapter; and
- a checksummed artifact manifest covering every file.

Transfer that directory without the full scratch result tree:

```bash
mkdir -p /Users/dhetting/src/final-g11-release
rsync -a --info=progress2 \
  kl1.hpc.nrel.gov:/scratch/dhetting/bsm_runs/${FINAL_ID}/publication_artifacts/ \
  /Users/dhetting/src/final-g11-release/publication_artifacts/
```

## Local zero-AU JDS release build

Use a new release root. `--replace` is necessary once to replace the tracked
internal-validation result macros and current figures in the manuscript
checkout. The installer will replace only the three generated LaTeX files, the
five cited figure PDFs, and its own install manifest.

```bash
cd /Users/dhetting/src/bsm-public-rf-g11-integration
pixi run python scripts/finalize_g11_manuscript_release.py \
  --publication-root /Users/dhetting/src/final-g11-release/publication_artifacts \
  --bsm-repo-root /Users/dhetting/src/bsm-public-rf-g11-integration \
  --manuscript-root /Users/dhetting/src/bsm-public-rf-manuscript \
  --release-root /Users/dhetting/src/final-g11-release/jds_release \
  --replace
```

The local build succeeds only if it creates 11 nontrivial SVGs, all five valid
article figure PDFs, an independent `PUBLICATION_ARTIFACT_AUDIT_PASS`, an
identity-bound manuscript install record, valid `manuscript.pdf` and
`coverpage.pdf`, a manuscript of no more than 25 pages, an exactly one-page
cover, no unresolved LaTeX citations or references, and
`JDS_RELEASE_BUILD_PASS.json`. The independent audit also verifies the numeric
standardized and raw-scale coefficient matrices, predictor standardization,
and raw-scale intercepts against the frozen model arrays; matching file hashes
alone are insufficient.

The same command creates two deterministic, internally tested archives in the
release root: `jds-submission-source.zip` contains the article/cover sources,
template assets, generated result surfaces, cited figures, and compiled PDFs;
`bsm-public-rf-supplement.zip` contains the complete checksummed publication
artifact bundle plus its independent audit, a top-level reproduction guide,
the repository license, and the pinned reproduction code/environment. Their
SHA-256 identities are recorded in `JDS_RELEASE_BUILD_PASS.json`; the
supplement filename is the fixed name used throughout the submission workflow.

## What may remain after this workflow

No additional HPC analysis is part of the manuscript plan. After the PASS
record, remaining work is editorial only: read the generated tables, remove
any prose contradicted by the observed results, perform the final anonymous JDS
format/readability check, and package the already-built files for submission.
If a scientific gate fails, the article's current scientific claim is not
supported; the controller correctly stops rather than spending the remaining
allocation on downstream work.
