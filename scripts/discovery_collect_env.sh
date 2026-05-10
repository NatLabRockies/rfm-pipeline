#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-artifacts/hpc_discovery}"
mkdir -p "$OUT_DIR"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "user=$(whoami)"
  echo "pwd=$(pwd)"
  echo "kernel=$(uname -a)"
} > "$OUT_DIR/system.txt"

{
  command -v scontrol >/dev/null 2>&1 && scontrol --version || true
  command -v sinfo >/dev/null 2>&1 && sinfo -o "%P %a %D %c %m %l %f" || true
  command -v sacctmgr >/dev/null 2>&1 && sacctmgr show qos format=name,maxwall,maxjobs,grptres,maxtresperjob -P || true
  command -v scontrol >/dev/null 2>&1 && scontrol show config || true
} > "$OUT_DIR/slurm_control.txt" 2>&1

{
  command -v lscpu >/dev/null 2>&1 && lscpu || true
  command -v free >/dev/null 2>&1 && free -h || true
  command -v numactl >/dev/null 2>&1 && numactl --hardware || true
  ulimit -a || true
  df -h || true
  df -ih || true
  mount || true
} > "$OUT_DIR/node_local.txt" 2>&1

env | sort > "$OUT_DIR/environment_vars.txt"

echo "Wrote artifacts to $OUT_DIR"
