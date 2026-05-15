#!/bin/bash
# Publication-ready full-dataset study: Step 3 — Monitor execution
#
# Purpose: Poll HPC for job status and progress through all stages
# Output: Stage summaries, completion status, artifact counts
# Time: ~1-2 minutes per check
#
# Usage: bash scripts/publication-run/03_monitor_publication_run.sh [interval]
# Example: bash scripts/publication-run/03_monitor_publication_run.sh 300  # Check every 5 minutes

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Optional interval (default 300 seconds = 5 minutes)
INTERVAL="${1:-300}"

echo "================================================================================"
echo "Publication-Ready Full-Dataset Study: MONITOR EXECUTION"
echo "================================================================================"
echo ""
echo "Orchestration config: configs/hpc/kestrel_publication_orchestration.yml"
echo "Check interval: $INTERVAL seconds"
echo ""
echo "Polling HPC for job status and stage progress..."
echo ""

# Run status check
pixi run hpc-workflow -- \
  --config configs/hpc/kestrel_publication_orchestration.yml \
  --action status

echo ""
echo "================================================================================"
echo "STATUS CHECK COMPLETE"
echo "================================================================================"
echo ""
echo "Interpretation guide:"
echo "  queued_or_pending     = Stage waiting to start or submitted"
echo "  running_or_waiting_reduce = Array jobs running, waiting for reduce"
echo "  completed             = Stage finished successfully"
echo "  failed                = Stage encountered errors (needs investigation)"
echo "  LOOKS_ACTIVE          = Jobs are running and progressing"
echo "  LOOKS_STALLED         = No recent progress (may need intervention)"
echo ""
echo "Expected stage sequence (each ~1-6 hours):"
echo "  1. output_conditioning (~1-2h)"
echo "  2. empirical_null_screening (~4-8h, high permutations)"
echo "  3. interaction_discovery array (~2-6h)"
echo "  4. interaction_discovery reduce (~1-2h)"
echo "  5. nonlinear_discovery (~4-12h)"
echo "  6. sparse_selection (~2-4h)"
echo "  7. final_artifacts (~2-4h)"
echo ""
echo "⏱️  Next check recommended in ~$((INTERVAL / 60)) minutes"
echo ""
echo "To collect results when all stages complete, run:"
echo "  bash scripts/publication-run/04_collect_publication_artifacts.sh"
echo ""
