#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=common_paths.sh
source "${REPO_ROOT}/scripts/kestrel/common_paths.sh"
DEFAULT_ARTIFACTS_ROOT="$(kestrel_default_artifacts_root "${REPO_ROOT}")"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-${DEFAULT_ARTIFACTS_ROOT}}"
LOGS_ROOT="${LOGS_ROOT:-$(kestrel_default_logs_root)}"

# Quick HPC test status snapshot (no looping)
# Shows: job state, % complete, any errors

echo "=== HPC Test Status $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo ""

# CPU 2-node test
echo "📊 CPU 2-node:"
{
  # Job status
  JOB_ID=$(squeue -u $USER --name="*bsm*cpu_scale_2*" --format="%i" --noheader 2>/dev/null | head -1)
  if [ -n "$JOB_ID" ]; then
    STATE=$(squeue -j $JOB_ID --format="%t" --noheader 2>/dev/null | head -1)
    echo "  State: $STATE (Job ID: $JOB_ID)"
  else
    # Check recent completion
    sacct -u $USER -S 2026-05-13 --format=JobID,State --noheader 2>/dev/null | grep -i "cpu_scale_2" | tail -1
  fi

  # Count completed shards (out of 2 total)
  COMPLETE=$(find "${ARTIFACTS_ROOT}/kestrel_cpu_scale_2_run/hpc_shards" -name "retained_interaction_pairs.csv" 2>/dev/null | wc -l)
  echo "  Progress: $COMPLETE / 2 shards complete ($(( COMPLETE * 50 ))%)"

  # Check for errors in logs
  if grep -q "error\|Error\|ERROR\|FAILED" "${LOGS_ROOT}/bsm_kestrel_cpu_scale_2/logs"/*.out 2>/dev/null; then
    echo "  ⚠️  Errors detected in logs"
    grep "error\|Error\|ERROR\|FAILED" "${LOGS_ROOT}/bsm_kestrel_cpu_scale_2/logs"/*.out 2>/dev/null | head -3
  fi
} 2>/dev/null || echo "  (no data yet)"
echo ""

# CPU 10-node test
echo "📊 CPU 10-node:"
{
  JOB_ID=$(squeue -u $USER --name="*bsm*cpu_scale_10*" --format="%i" --noheader 2>/dev/null | head -1)
  if [ -n "$JOB_ID" ]; then
    STATE=$(squeue -j $JOB_ID --format="%t" --noheader 2>/dev/null | head -1)
    echo "  State: $STATE (Job ID: $JOB_ID)"
  else
    sacct -u $USER -S 2026-05-13 --format=JobID,State --noheader 2>/dev/null | grep -i "cpu_scale_10" | tail -1
  fi

  COMPLETE=$(find "${ARTIFACTS_ROOT}/kestrel_cpu_scale_10_run/hpc_shards" -name "retained_interaction_pairs.csv" 2>/dev/null | wc -l)
  echo "  Progress: $COMPLETE / 10 shards complete ($(( COMPLETE * 10 ))%)"

  if grep -q "error\|Error\|ERROR\|FAILED" "${LOGS_ROOT}/bsm_kestrel_cpu_scale_10/logs"/*.out 2>/dev/null; then
    echo "  ⚠️  Errors detected in logs"
  fi
} 2>/dev/null || echo "  (no data yet)"
echo ""

# CPU 1000-node test
echo "📊 CPU 1000-node:"
{
  JOB_ID=$(squeue -u $USER --name="*bsm*cpu_scale_1000*" --format="%i" --noheader 2>/dev/null | head -1)
  if [ -n "$JOB_ID" ]; then
    STATE=$(squeue -j $JOB_ID --format="%t" --noheader 2>/dev/null | head -1)
    echo "  State: $STATE (Job ID: $JOB_ID)"
  else
    sacct -u $USER -S 2026-05-13 --format=JobID,State --noheader 2>/dev/null | grep -i "cpu_scale_1000" | tail -1
  fi

  COMPLETE=$(find "${ARTIFACTS_ROOT}/kestrel_cpu_scale_1000_run/hpc_shards" -name "retained_interaction_pairs.csv" 2>/dev/null | wc -l)
  PERCENT=$(( COMPLETE / 10 ))  # ~1000 shards, so divide count by 10 for rough percentage
  echo "  Progress: $COMPLETE / 1000 shards complete (~$PERCENT%)"

  if grep -q "error\|Error\|ERROR\|FAILED" "${LOGS_ROOT}/bsm_kestrel_cpu_scale_1000/logs"/*.out 2>/dev/null; then
    echo "  ⚠️  Errors detected in logs"
  fi
} 2>/dev/null || echo "  (no data yet)"
echo ""

# GPU test
echo "📊 GPU H100:"
{
  JOB_ID=$(squeue -u $USER --name="*bsm*gpu*h100*" --format="%i" --noheader 2>/dev/null | head -1)
  if [ -n "$JOB_ID" ]; then
    STATE=$(squeue -j $JOB_ID --format="%t" --noheader 2>/dev/null | head -1)
    REASON=$(squeue -j $JOB_ID --format="%.20R" --noheader 2>/dev/null | head -1)
    echo "  State: $STATE (Job ID: $JOB_ID)"
    if [ "$STATE" = "PD" ]; then
      echo "  Reason: $REASON"
    fi
  else
    sacct -u $USER -S 2026-05-13 --format=JobID,State --noheader 2>/dev/null | grep -i "gpu_h100" | tail -1
  fi

  COMPLETE=$(find "${ARTIFACTS_ROOT}/kestrel_gpu_h100_run/hpc_shards" -name "retained_interaction_pairs.csv" 2>/dev/null | wc -l)
  echo "  Progress: $COMPLETE / 10 shards complete ($(( COMPLETE * 10 ))%)"

  if grep -q "error\|Error\|ERROR\|FAILED" "${LOGS_ROOT}/bsm_kestrel_gpu_h100/logs"/*.out 2>/dev/null; then
    echo "  ⚠️  Errors detected in logs"
  fi
} 2>/dev/null || echo "  (no data yet)"
echo ""

echo "Legend:"
echo "  R  = Running"
echo "  PD = Pending (queued)"
echo "  CA = Cancelled"
echo "  ST = Stopped"
echo ""
echo "To see details: squeue -u \$USER"
echo "Artifacts root: ${ARTIFACTS_ROOT}"
echo "Logs root: ${LOGS_ROOT}"
echo "To tail logs: tail -f ${LOGS_ROOT}/bsm_kestrel_cpu_scale_*/logs/*.out"
