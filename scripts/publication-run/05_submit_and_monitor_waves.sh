#!/bin/bash
# Publication-ready full-dataset study: Submit and monitor in 4-hour waves
#
# Submits jobs in batches, monitors each wave to completion, then submits next wave.
# This keeps the queue flowing without overwhelming it.
#
# Usage: bash scripts/publication-run/05_submit_and_monitor_waves.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================================"
echo "Publication-Ready Full-Dataset Study: WAVE SUBMISSION & MONITORING"
echo "================================================================================"
echo ""
echo "Strategy: Submit 4-hour jobs in waves, monitor each to completion"
echo "Orchestration config: configs/hpc/kestrel_publication_orchestration.yml"
echo "Pipeline config: configs/hpc/kestrel_publication_full_dataset.yml"
echo "Dataset: real_full_dataset (30,000 samples, 160 inputs, 23,497 outputs)"
echo ""

# Configuration
MONITOR_INTERVAL=60          # Check status every 60 seconds
POLL_TIMEOUT=14400           # 4 hours = 14400 seconds
START_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
WAVE=1

echo "Start time: $START_TIME"
echo "Each wave target: <4 hours"
echo "Monitor interval: $MONITOR_INTERVAL seconds"
echo ""
echo "================================================================================"

while true; do
  echo ""
  echo "🌊 WAVE $WAVE — Submitting batch..."
  echo ""

  # Submit jobs
  bash scripts/publication-run/02_live_submission.sh 2>&1 | tail -20

  echo ""
  echo "Waiting for jobs to complete (~4 hours)..."
  echo "You can interrupt (Ctrl+C) to stop monitoring, jobs will continue running."
  echo ""

  # Extract job IDs from the output for polling (for now, just use status check)
  MONITOR_START=$(date +%s)
  POLL_COUNT=0

  while true; do
    ELAPSED=$(($(date +%s) - MONITOR_START))
    POLL_COUNT=$((POLL_COUNT + 1))

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Poll #$POLL_COUNT (elapsed: ${ELAPSED}s)"

    # Check job status
    STATUS_OUT=$(bash scripts/publication-run/03_monitor_publication_run.sh 2>&1 | tail -30)
    echo "$STATUS_OUT" | head -15

    # Look for completion indicators
    if echo "$STATUS_OUT" | grep -q "all.*completed\|LOOKS_COMPLETE\|100%"; then
      echo ""
      echo "✅ Wave $WAVE complete!"
      break
    fi

    # Check if jobs exist at all (all done)
    ACTIVE_JOBS=$(ssh -T dhetting@kl1.hpc.nrel.gov "squeue -u dhetting | wc -l" 2>/dev/null || echo "0")
    if [ "$ACTIVE_JOBS" -lt 2 ]; then
      echo ""
      echo "✅ No active jobs. Wave $WAVE complete!"
      break
    fi

    # Timeout check (4+ hours)
    if [ $ELAPSED -gt $POLL_TIMEOUT ]; then
      echo ""
      echo "⏱️  4-hour window elapsed. Assuming wave complete (may need manual verification)."
      break
    fi

    echo "Waiting ${MONITOR_INTERVAL}s until next check..."
    sleep "$MONITOR_INTERVAL"
  done

  WAVE=$((WAVE + 1))

  # Ask if user wants to continue
  echo ""
  echo "================================================================================"
  echo "Wave $WAVE ready to submit. Continue? (yes/no)"
  read -r CONTINUE
  if [ "$CONTINUE" != "yes" ] && [ "$CONTINUE" != "y" ]; then
    echo ""
    echo "✓ Exiting. Jobs may still be running on HPC."
    echo "  Monitor with: bash scripts/publication-run/03_monitor_publication_run.sh"
    break
  fi
done

echo ""
echo "================================================================================"
echo "✓ WAVE SUBMISSION COMPLETE"
echo "================================================================================"
echo ""
echo "When all waves complete, collect results:"
echo "  bash scripts/publication-run/04_collect_publication_artifacts.sh"
echo ""
