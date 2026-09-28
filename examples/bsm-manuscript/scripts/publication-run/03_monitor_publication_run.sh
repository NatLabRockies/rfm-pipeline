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

echo "RETIRED: use the content-addressed G11 campaign package; this legacy workflow is non-executable." >&2
exit 64

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Optional polling interval in seconds. When unset → single status check (one-shot).
# When set → poll every N seconds until Ctrl-C.
INTERVAL="${1:-}"

run_status_check() {
  echo "================================================================================"
  echo "Publication-Ready Full-Dataset Study: MONITOR EXECUTION ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo "================================================================================"
  echo ""
  echo "Orchestration config: configs/hpc/kestrel_publication_orchestration.yml"
  if [[ -n "$INTERVAL" ]]; then
    echo "Mode: poll every ${INTERVAL}s (Ctrl-C to stop)"
  else
    echo "Mode: one-shot status check"
  fi
  echo ""

  pixi run hpc-workflow -- \
    --config configs/hpc/kestrel_publication_orchestration.yml \
    --action status

  echo ""
  echo "Interpretation guide:"
  echo "  queued_or_pending         = Stage waiting to start or submitted"
  echo "  running_or_waiting_reduce = Array jobs running, waiting for reduce"
  echo "  completed                 = Stage finished successfully"
  echo "  failed                    = Stage encountered errors (needs investigation)"
  echo "  LOOKS_ACTIVE              = Jobs are running and progressing"
  echo "  LOOKS_STALLED             = No recent progress (may need intervention)"
  echo ""
  echo "Expected stage sequence (each ~1-6 hours):"
  echo "  1. output_conditioning (~1-2h)"
  echo "  2. empirical_null_screening (~4-8h, high permutations)"
  echo "  3. interaction_discovery, array + reduce (~3-8h)"
  echo "  4. nonlinear_discovery (~4-12h)"
  echo "  5. sparse_selection (~2-4h)"
  echo "  6. final_manuscript_artifacts (~2-4h)"
  echo ""
  echo "To collect results when all stages complete, run:"
  echo "  bash scripts/publication-run/04_collect_publication_artifacts.sh"
  echo ""
}

if [[ -n "$INTERVAL" ]]; then
  trap 'echo ""; echo "Polling stopped."; exit 0' INT
  while true; do
    run_status_check
    echo "⏱️  Next check in ${INTERVAL}s (Ctrl-C to stop)"
    sleep "$INTERVAL"
  done
else
  run_status_check
fi
