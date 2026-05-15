#!/bin/bash
# Publication-Ready Full-Dataset Study — Quick Reference

cat << 'EOF'

╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║        Publication-Ready Full-Dataset Study — Quick Start         ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

DATASET
  • Real BSM study (30,000 samples, 160 inputs, 23,497 outputs)
  • Location: artifacts/preprocessed_real_data/
  • Status: ✓ Ready

CONFIGURATION
  • File: configs/hpc/kestrel_publication_full_dataset.yml
  • Hyperparameters: Quality-first (201 perms, 250 trees, 100 bootstrap)
  • Status: ✓ Validated

SCRIPTS
  Location: scripts/publication-run/

  OPTION 1: Automated (Recommended)
  ─────────────────────────────────
  bash scripts/publication-run/00_publication_run_master.sh

  Walks through all 4 stages with prompts.
  Estimated time: 24+ hours (with monitoring pauses)

  OPTION 2: Individual Stages (Manual Control)
  ─────────────────────────────────────────────

  Stage 1: Dry-run validation (~30 seconds)
  bash scripts/publication-run/01_dry_run_submission.sh
  ✓ Generates and validates SLURM scripts without submitting

  Stage 2: Live submission (~2 minutes)
  bash scripts/publication-run/02_live_submission.sh
  ✓ Submits to HPC, returns job IDs

  Stage 3: Monitor execution (~1-2 min per check)
  bash scripts/publication-run/03_monitor_publication_run.sh
  bash scripts/publication-run/03_monitor_publication_run.sh 300  # Auto-check every 5 min
  ✓ Polls HPC for progress through all 7 stages (~24 hours total)

  Stage 4: Collect results (~10 minutes)
  bash scripts/publication-run/04_collect_publication_artifacts.sh
  ✓ Pulls final package from HPC to ./artifacts/publication_full_dataset_results/

EXPECTED STAGES (each 1-6 hours)
  1. output_conditioning           (~1-2 hours)
  2. empirical_null_screening      (~4-8 hours) [high permutations]
  3. interaction_discovery array   (~2-6 hours)
  4. interaction_discovery reduce  (~1-2 hours)
  5. nonlinear_discovery           (~4-12 hours)
  6. sparse_selection              (~2-4 hours)
  7. final_artifacts               (~2-4 hours)

  Total: ~24 hours

STATUS INTERPRETATION
  queued_or_pending     = Waiting to start
  running_or_waiting    = Jobs running
  completed             = Stage done ✓
  failed                = Needs investigation ✗
  LOOKS_ACTIVE          = Progress happening
  LOOKS_STALLED         = No recent progress

QUICK START
─────────────────────────────────────────────────

1. VALIDATE (30 seconds)
   bash scripts/publication-run/01_dry_run_submission.sh

   Check output for:
   ✓ Valid SLURM headers
   ✓ Correct partition (shared)
   ✓ Correct walltime (24:00:00)
   ✓ All stage commands present

2. SUBMIT (2 minutes)
   bash scripts/publication-run/02_live_submission.sh

   Check output for:
   ✓ "HPC workflow action complete" message
   ✓ No SSH errors
   ✓ Tracking file created

3. MONITOR (Every 2-4 hours, ~24 hours total)
   bash scripts/publication-run/03_monitor_publication_run.sh

   OR automated (checks every 5 minutes):
   bash scripts/publication-run/03_monitor_publication_run.sh 300

   Check output for:
   ✓ All stages progressing
   ✓ No "failed" status
   ✓ Eventually all showing "completed"

4. COLLECT (10 minutes)
   bash scripts/publication-run/04_collect_publication_artifacts.sh

   Check output for:
   ✓ "All critical artifacts present"
   ✓ Results in ./artifacts/publication_full_dataset_results/
   ✓ Manifest and figures extracted

OUTPUTS
────────

Local directory: ./artifacts/publication_full_dataset_results/
Contains:
  • manifest/              - Reproducibility, configs, commands
  • tables/                - CSV results and statistics
  • figures/               - Publication-ready SVG figures
  • logs/                  - Execution logs from each stage

File locations:
  • Manifest:        manifest/hpc_run_manifest.json
  • Summary:         manifest/run_summary.csv
  • Commands trace:  manifest/commands.json
  • Reproduction:    manifest/reproduction_recipe.md
  • Figures:         figures/*.svg
  • Tables:          tables/*.csv

TROUBLESHOOTING
───────────────

Issue: Dry-run fails with config error
→ pixi run python -c "from bsm_rfm.config import load_config; load_config('configs/hpc/kestrel_publication_full_dataset.yml')"

Issue: Submit hangs
→ ssh -T dhetting@kl1.hpc.nrel.gov "echo 'SSH works'"

Issue: Monitor shows "queued" after 30+ minutes
→ ssh -T dhetting@kl1.hpc.nrel.gov "squeue -u dhetting"

Issue: Collection fails
→ bash scripts/publication-run/03_monitor_publication_run.sh
→ Verify all stages show "completed"

MORE DETAILS
─────────────

Read the full documentation:
cat scripts/publication-run/README.md

View individual stage scripts:
ls -la scripts/publication-run/

List available configurations:
ls -l configs/hpc/kestrel_*.yml

═══════════════════════════════════════════════════════════════════

Ready to start?

  bash scripts/publication-run/00_publication_run_master.sh

═══════════════════════════════════════════════════════════════════

EOF
