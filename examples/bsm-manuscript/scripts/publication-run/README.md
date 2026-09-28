# Publication-Ready Full-Dataset Study — Script Suite

## Overview

This directory contains a complete script suite for executing the publication-grade full-dataset study on NREL Kestrel HPC. The workflow has 4 independent steps that can be run sequentially or individually.

## Dataset

- **Source**: Real BSM study data (30,000 samples)
- **Inputs**: 160 parameters (raw, not standardized)
- **Outputs**: 23,495 economic metrics
- **Location**: `artifacts/preprocessed_real_data_30k/`
- **Size**: 2.1 GB

## Configuration

- **Config file**: `configs/hpc/kestrel_publication_full_dataset.yml`
- **Hyperparameters**: Quality-first (201 permutations, 250 trees, 100 bootstrap)
- **HPC resource**: Shared partition, 4-hour window per HPC job; full pipeline runs 6 stages (~24h total wall time), 240 GB memory, 104 CPUs
- **Expected total runtime**: ~24 hours

## Scripts

### Option 1: Automated Orchestration (Recommended)

```bash
bash scripts/publication-run/00_publication_run_master.sh
```

**What it does:**

- Prompts through all 4 steps sequentially
- Validates before proceeding to next step
- Provides clear instructions and interpretations
- Estimated time: 24+ hours (with monitoring pauses)

______________________________________________________________________

### Option 2: Individual Scripts (Manual Control)

#### Step 1: Dry-run Validation

```bash
bash scripts/publication-run/01_dry_run_submission.sh
```

**Purpose**: Generate SLURM scripts without submitting to HPC

> **Prerequisite**: dry-run still SSHes to the configured remote host and
> runs `rfm-hpc-submit --dry-run` there. SLURM scripts are generated on
> the remote scratch dir (not locally), so a working SSH connection and
> a current remote checkout are required. To skip SSH entirely and only
> print the local plan, pass `--generate-only` to `hpc_workflow.py`.

**Outputs**:

- Validates script syntax
- Checks partition, walltime, memory settings
- Confirms the entry-stage command is present

**Time**: ~30 seconds

**Next step**: Review output, then run Step 2

______________________________________________________________________

#### Step 2: Live Submission

```bash
bash scripts/publication-run/02_live_submission.sh
```

**Purpose**: Submit publication-grade job to Kestrel HPC

**Outputs**:

- Job submission confirmation
- Job IDs (if available)
- Tracking file saved to `artifacts/publication_run_tracking.txt`

**Time**: ~2 minutes

**What to check**:

- Look for "HPC workflow action complete" message
- Note any job IDs in output
- Verify no error messages

**Next step**: Wait for jobs to start (5-15 minutes), then run Step 3

______________________________________________________________________

#### Step 3: Monitor Execution

```bash
# Check status once
bash scripts/publication-run/03_monitor_publication_run.sh

# Check status every 5 minutes (automated)
bash scripts/publication-run/03_monitor_publication_run.sh 300

# Check status every 15 minutes (slower polling)
bash scripts/publication-run/03_monitor_publication_run.sh 900
```

**Purpose**: Poll HPC for job status and stage progress

**Outputs**:

- Current stage status (queued, running, completed, failed)
- Job health indicators
- Artifact counts per stage
- Log file locations

**Time**: ~1-2 minutes per check

**Expected stage sequence** (each 1-6 hours):

1. output_conditioning (1-2 hours)
1. empirical_null_screening (4-8 hours, high permutations)
1. interaction_discovery, array + reduce (3-8 hours)
1. nonlinear_discovery (4-12 hours)
1. sparse_selection (2-4 hours)
1. final_manuscript_artifacts (2-4 hours)

**Status interpretation**:

- `queued_or_pending`: Stage waiting to start
- `running_or_waiting_reduce`: Array jobs running
- `completed`: Stage finished successfully
- `LOOKS_ACTIVE`: Progress is happening
- `LOOKS_STALLED`: No recent progress (investigate)

**When to run**: Every 2-4 hours during the 24-hour window

______________________________________________________________________

#### Step 4: Collect Artifacts

```bash
bash scripts/publication-run/04_collect_publication_artifacts.sh
```

**Purpose**: Pull final results from HPC to local machine

**Outputs**:

- Complete study package at `artifacts/publication_full_dataset_results/`
- Includes:
  - Reproducibility manifest (commands, configs, versions)
  - Stage-by-stage execution summary
  - Publication tables and figures
  - Statistical summaries

**Time**: ~10 minutes

**What to check**:

- Look for "✓ All critical artifacts present"
- Verify directory structure in extracted bundle
- Check manifest files are non-empty

**Next step**: Review results, prepare for manuscript integration

______________________________________________________________________

## Quick Start

### First Time Setup (Validate Configuration)

```bash
# 1. Dry-run to validate scripts
bash scripts/publication-run/01_dry_run_submission.sh

# 2. If dry-run looks good, submit
bash scripts/publication-run/02_live_submission.sh

# 3. Save start time locally
date -u > artifacts/publication_run_start_time.txt
```

### During Execution (Monitoring)

```bash
# Check status periodically (every few hours)
bash scripts/publication-run/03_monitor_publication_run.sh

# Look for all stages showing "completed"
# Expected: all 6 stages done in ~24 hours
```

### After Execution (Collect Results)

```bash
# Pull results from HPC
bash scripts/publication-run/04_collect_publication_artifacts.sh

# Results available at:
ls -R artifacts/publication_full_dataset_results/
```

______________________________________________________________________

## Troubleshooting

### Issue: Dry-run validation fails

**Solution**: Check `configs/hpc/kestrel_publication_full_dataset.yml` syntax

```bash
pixi run python -c "from rfm_pipeline.config import load_config; load_config('configs/hpc/kestrel_publication_full_dataset.yml')"
```

### Issue: Live submission hangs or fails

**Solution**: Verify SSH access to Kestrel

```bash
ssh -T ${USER}@${HPC_HOST} "echo 'SSH works'"
```

### Issue: Monitor script shows "queued_or_pending" after 30+ minutes

**Solution**: Check HPC queue status manually

```bash
ssh -T ${USER}@${HPC_HOST} "squeue -u ${USER}"
```

### Issue: Collection fails with "no bundle found"

**Solution**: Verify stages actually completed (not just queued)

```bash
bash scripts/publication-run/03_monitor_publication_run.sh
# Look for "completed" status on all 6 stages
```

______________________________________________________________________

## Environment Variables

These helper scripts do **not** read environment-variable overrides; the
config path and output directory are hard-coded to the publication-run
values (`configs/hpc/kestrel_publication_orchestration.yml` and the
`local_bundle_dir` declared inside it). To run with a different config or
collect into a different directory, either edit the orchestration YAML or
invoke `pixi run hpc-workflow -- --config <your-config> --action <submit|status|collect>`
directly.

______________________________________________________________________

## Expected Outputs

### From Dry-run (Step 1)

- SLURM script snippets printed to console
- No files created on HPC

### From Submission (Step 2)

- `artifacts/publication_run_tracking.txt` (local tracking file)
- Remote job IDs (noted in script output)

### From Monitoring (Step 3)

- Status summaries printed to console
- No local files created (remote read-only)

### From Collection (Step 4)

- `artifacts/publication_full_dataset_results/` (complete study package)
  - `manifest/` (metadata and reproducibility)
  - `tables/` (CSV tables and statistics)
  - `figures/` (SVG publication figures)
  - `logs/` (execution logs from each step)

______________________________________________________________________

## File Structure

```
scripts/publication-run/
├── 00_publication_run_master.sh        # Orchestrator (run this first)
├── 01_dry_run_submission.sh            # Validate scripts
├── 02_live_submission.sh               # Submit to HPC
├── 03_monitor_publication_run.sh       # Check progress
├── 04_collect_publication_artifacts.sh # Pull results
└── README.md                           # This file
```

______________________________________________________________________

## Success Criteria

### Step 1 (Dry-run) ✓

- Script generation succeeds without errors
- SLURM headers are valid
- No warnings about missing configs

### Step 2 (Submission) ✓

- "HPC workflow action complete" message appears
- No SSH or permission errors
- Tracking file created locally

### Step 3 (Monitoring) ✓

- All 6 stages show "queued_or_pending" initially
- Stages gradually progress to "running_or_waiting_reduce"
- All stages show "completed" after ~24 hours

### Step 4 (Collection) ✓

- Bundle extracted successfully
- "All critical artifacts present" message appears
- Study package in `artifacts/publication_full_dataset_results/`

______________________________________________________________________

## Support

For issues or questions:

1. Review the troubleshooting section above
1. Check HPC status manually: `squeue -u ${USER}`
1. Review remote logs: `tail -f /scratch/${USER}/bsm/bsm_kestrel_publication_full/logs/*.out`
1. Check latest bundle: `ls -lh artifacts/kestrel_collected_bundles/ | tail -1`

______________________________________________________________________

**Ready to start?**

```bash
bash scripts/publication-run/00_publication_run_master.sh
```
