#!/usr/bin/env bash
set -euo pipefail

# Comprehensive HPC test status and diagnostics collector
# Runs on Kestrel; outputs to scratch for scp back to local machine

DIAG_DIR="/scratch/dhetting/hpc_diagnostics_$(date +%s)"
mkdir -p "$DIAG_DIR"

echo "Collecting HPC test diagnostics to $DIAG_DIR"

# Job status history
echo "=== SLURM Job History (last 24h) ===" > "$DIAG_DIR/01_sacct_all.txt"
sacct -u $USER -S $(date -d '24 hours ago' +%Y-%m-%d) \
  --format=JobID,JobName,State,ExitCode,Elapsed,Start,End >> "$DIAG_DIR/01_sacct_all.txt" 2>&1 || true

# Current queue
echo "=== Current Queue ===" > "$DIAG_DIR/02_squeue.txt"
squeue -u $USER >> "$DIAG_DIR/02_squeue.txt" 2>&1 || true

# CPU 2-node diagnostics
echo "=== CPU 2-node Test ===" > "$DIAG_DIR/03_cpu_2_status.txt"
{
  echo "Job history:"
  sacct -u $USER -S 2026-05-13 | grep -i "cpu_scale_2" || echo "  (no jobs found)"
  echo ""
  echo "Artifacts produced:"
  find artifacts/kestrel_cpu_scaling_suite/cpu_nodes_2 -type f -name "*.csv" | head -20 || echo "  (none)"
  echo ""
  echo "Latest log file:"
  ls -lht /scratch/dhetting/bsm/bsm_kestrel_cpu_scale_2/logs/*.out 2>/dev/null | head -1 || echo "  (not found)"
} >> "$DIAG_DIR/03_cpu_2_status.txt"

# CPU 2-node logs
find /scratch/dhetting/bsm/bsm_kestrel_cpu_scale_2/logs -name "*.out" -type f 2>/dev/null \
  | head -3 | while read f; do
    echo "--- $(basename $f) ---" >> "$DIAG_DIR/03_cpu_2_status.txt"
    tail -50 "$f" >> "$DIAG_DIR/03_cpu_2_status.txt" 2>&1 || true
  done

# CPU 10-node diagnostics
echo "=== CPU 10-node Test ===" > "$DIAG_DIR/04_cpu_10_status.txt"
{
  echo "Job history:"
  sacct -u $USER -S 2026-05-13 | grep -i "cpu_scale_10" || echo "  (no jobs found)"
  echo ""
  echo "Artifacts produced:"
  find artifacts/kestrel_cpu_scaling_suite/cpu_nodes_10 -type f -name "*.csv" | head -20 || echo "  (none)"
  echo ""
  echo "Latest log file:"
  ls -lht /scratch/dhetting/bsm/bsm_kestrel_cpu_scale_10/logs/*.out 2>/dev/null | head -1 || echo "  (not found)"
} >> "$DIAG_DIR/04_cpu_10_status.txt"

# CPU 10-node logs
find /scratch/dhetting/bsm/bsm_kestrel_cpu_scale_10/logs -name "*.out" -type f 2>/dev/null \
  | head -3 | while read f; do
    echo "--- $(basename $f) ---" >> "$DIAG_DIR/04_cpu_10_status.txt"
    tail -50 "$f" >> "$DIAG_DIR/04_cpu_10_status.txt" 2>&1 || true
  done

# CPU 1000-node diagnostics
echo "=== CPU 1000-node Test ===" > "$DIAG_DIR/05_cpu_1000_status.txt"
{
  echo "Job history:"
  sacct -u $USER -S 2026-05-13 | grep -i "cpu_scale_1000" || echo "  (no jobs found)"
  echo ""
  echo "Artifacts produced:"
  find artifacts/kestrel_cpu_scaling_suite/cpu_nodes_1000 -type f -name "*.csv" | wc -l
  echo ""
  echo "Latest log file:"
  ls -lht /scratch/dhetting/bsm/bsm_kestrel_cpu_scale_1000/logs/*.out 2>/dev/null | head -1 || echo "  (not found)"
} >> "$DIAG_DIR/05_cpu_1000_status.txt"

# CPU 1000-node logs
find /scratch/dhetting/bsm/bsm_kestrel_cpu_scale_1000/logs -name "*.out" -type f 2>/dev/null \
  | head -3 | while read f; do
    echo "--- $(basename $f) ---" >> "$DIAG_DIR/05_cpu_1000_status.txt"
    tail -50 "$f" >> "$DIAG_DIR/05_cpu_1000_status.txt" 2>&1 || true
  done

# GPU diagnostics
echo "=== GPU Test ===" > "$DIAG_DIR/06_gpu_status.txt"
{
  echo "Job history:"
  sacct -u $USER -S 2026-05-13 | grep -i "gpu_h100" || echo "  (no jobs found)"
  echo ""
  echo "Artifacts produced:"
  find artifacts/kestrel_gpu_h100_run -type f -name "*.csv" | head -20 || echo "  (none)"
  echo ""
  echo "Latest log file:"
  ls -lht /scratch/dhetting/bsm/bsm_kestrel_gpu_h100/logs/*.out 2>/dev/null | head -1 || echo "  (not found)"
} >> "$DIAG_DIR/06_gpu_status.txt"

# GPU logs
find /scratch/dhetting/bsm/bsm_kestrel_gpu_h100/logs -name "*.out" -type f 2>/dev/null \
  | head -3 | while read f; do
    echo "--- $(basename $f) ---" >> "$DIAG_DIR/06_gpu_status.txt"
    tail -50 "$f" >> "$DIAG_DIR/06_gpu_status.txt" 2>&1 || true
  done

# Copy any .err files if they exist
mkdir -p "$DIAG_DIR/error_files"
find /scratch/dhetting/bsm -name "*.err" -type f 2>/dev/null | head -10 | xargs -I {} cp {} "$DIAG_DIR/error_files/" 2>/dev/null || true

echo ""
echo "Diagnostics collected to: $DIAG_DIR"
echo ""
echo "To download locally, run:"
echo "  scp -r kl1.hpc.nrel.gov:$DIAG_DIR ~/hpc_diagnostics"
echo ""
echo "Summary:"
ls -lh "$DIAG_DIR"
