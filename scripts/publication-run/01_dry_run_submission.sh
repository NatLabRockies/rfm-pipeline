#!/bin/bash
# Publication-ready full-dataset study: Step 1 — Dry-run submission
#
# Purpose: Validate SLURM script generation without submitting to HPC
# Output: Generated scripts in /tmp/ for review
# Time: ~30 seconds
#
# Usage: bash scripts/publication-run/01_dry_run_submission.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================================"
echo "Publication-Ready Full-Dataset Study: DRY-RUN SUBMISSION"
echo "================================================================================"
echo ""
echo "Orchestration config: configs/hpc/kestrel_publication_orchestration.yml"
echo "Pipeline config: configs/hpc/kestrel_publication_full_dataset.yml"
echo "Dataset: real_full_dataset (30,000 samples)"
echo "Mode: DRY-RUN (scripts generated but NOT submitted)"
echo ""
echo "Starting dry-run validation..."
echo ""

# Run dry-run
pixi run hpc-workflow -- \
  --config configs/hpc/kestrel_publication_orchestration.yml \
  --action submit \
  --dry-run

echo ""
echo "================================================================================"
echo "✓ DRY-RUN COMPLETE"
echo "================================================================================"
echo ""
echo "Review the generated scripts above for:"
echo "  ✓ Valid SLURM headers (#SBATCH)"
echo "  ✓ Correct partition (shared)"
echo "  ✓ Correct walltime (24:00:00)"
echo "  ✓ Correct memory (240 GB)"
echo "  ✓ All stage commands present"
echo ""
echo "Next step: bash scripts/publication-run/02_live_submission.sh"
echo ""
