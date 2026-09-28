#!/bin/bash
# Publication-ready full-dataset study: Step 2 — Live submission
#
# Purpose: Submit publication-grade run to HPC with job tracking
# Output: Job IDs, remote logs, status snapshot
# Time: ~2 minutes
#
# Usage: bash scripts/publication-run/02_live_submission.sh

set -euo pipefail

echo "RETIRED: use the content-addressed G11 campaign package; this legacy workflow is non-executable." >&2
exit 64

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================================"
echo "Publication-Ready Full-Dataset Study: LIVE SUBMISSION"
echo "================================================================================"
echo ""
echo "Orchestration config: configs/hpc/kestrel_publication_orchestration.yml"
echo "Pipeline config: configs/hpc/kestrel_publication_full_dataset.yml"
echo "Dataset: real_full_dataset (30,000 samples, 160 inputs, 23,495 outputs)"
echo "Hyperparameters: Quality-first (201 null perms, 250 trees, 100 bootstrap)"
echo "Walltime: 04:00:00 per job stage | ~24h total across 6 pipeline stages | Partition: shared | Memory: 240 GB"
echo ""
echo "⚠️  SUBMITTING TO HPC NOW..."
echo ""

# Capture start time
START_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Run live submission (not dry-run)
# Note: uses orchestration config which points to pipeline config
SUBMIT_OUTPUT=$(pixi run hpc-workflow -- \
  --config configs/hpc/kestrel_publication_orchestration.yml \
  --action submit 2>&1) || { rc=$?; printf '%s\n' "$SUBMIT_OUTPUT"; exit "$rc"; }

echo "$SUBMIT_OUTPUT"

# Extract and save job IDs if present
echo ""
echo "================================================================================"
echo "✓ SUBMISSION COMPLETE"
echo "================================================================================"
echo ""
echo "Start time: $START_TIME"
echo "Expected completion: ~24 hours from submission"
echo ""
echo "To track progress, run:"
echo "  bash scripts/publication-run/03_monitor_publication_run.sh"
echo ""
echo "To collect results when complete, run:"
echo "  bash scripts/publication-run/04_collect_publication_artifacts.sh"
echo ""

# Create a simple tracking file
TRACKING_FILE="./artifacts/publication_run_tracking.txt"
cat > "$TRACKING_FILE" << EOF
=== Publication-Ready Full-Dataset Study Tracking ===
Start time (UTC): $START_TIME
Config: configs/hpc/kestrel_publication_full_dataset.yml
Dataset: real_full_dataset (30k samples)
Submission output:
$SUBMIT_OUTPUT
EOF

echo "Tracking info saved to: $TRACKING_FILE"
echo ""
