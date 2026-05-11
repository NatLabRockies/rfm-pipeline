#!/usr/bin/env bash
# Shared Kestrel/NLR defaults and helpers for BSM HPC discovery scripts.
# Source this file from scripts rather than editing each script.

# Allocation/project handle requested by the user.
export NLR_ACCOUNT="${NLR_ACCOUNT:-bsm}"

# Storage roots. Keep software/envs and durable results in ProjectFS; keep large
# transient job data in ScratchFS unless a real node-local TMPDIR is available.
export KESTREL_PROJECT_ROOT="${KESTREL_PROJECT_ROOT:-/projects/${NLR_ACCOUNT}}"
export KESTREL_SCRATCH_ROOT="${KESTREL_SCRATCH_ROOT:-/scratch/${USER}/${NLR_ACCOUNT}_hpc_discovery}"

# User-requested queues: CPU discovery in debug; GPU discovery in short H100 GPU queue.
export KESTREL_CPU_PARTITION="${KESTREL_CPU_PARTITION:-debug}"
export KESTREL_GPU_PARTITION="${KESTREL_GPU_PARTITION:-gpu-h100s}"

# Default artifact location can be overridden per script with the first argument
# or DISCOVERY_OUT_DIR.
export DISCOVERY_OUT_DIR="${DISCOVERY_OUT_DIR:-${PWD}/artifacts/hpc_discovery}"

# Project-scoped Pixi/cache paths. Avoid /home for heavy environment metadata.
export PIXI_HOME="${PIXI_HOME:-${KESTREL_PROJECT_ROOT}/.pixi}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-${KESTREL_PROJECT_ROOT}/.cache/pixi}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${KESTREL_PROJECT_ROOT}/.cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${KESTREL_PROJECT_ROOT}/.conda/pkgs}"
export PATH="${PIXI_HOME}/bin:${PATH}"

# Conservative default threading for Python jobs. Individual jobs can override.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

# Monitor output controls. Keep default output useful without flooding the terminal.
export MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-10}"
export MONITOR_TAIL_LINES="${MONITOR_TAIL_LINES:-8}"

kestrel_timestamp_utc() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

kestrel_status() {
  local message="$1"
  echo "status_timestamp=$(kestrel_timestamp_utc) ${message}"
}

kestrel_mkdir() {
  mkdir -p "$1"
}

kestrel_have_partition() {
  local partition="$1"
  command -v sinfo >/dev/null 2>&1 || return 1
  sinfo -h -p "$partition" -o '%P' 2>/dev/null | grep -q .
}

kestrel_realpath() {
  # realpath exists on Kestrel, but keep a fallback for portability.
  if command -v realpath >/dev/null 2>&1; then
    realpath "$1"
  else
    python3 - <<PY
from pathlib import Path
print(Path('$1').expanduser().resolve())
PY
  fi
}

kestrel_safe_name() {
  printf '%s' "$1" | tr '/: ' '___' | tr -cd 'A-Za-z0-9._-'
}

kestrel_is_real_disk_tmpdir() {
  # Return success only when TMPDIR exists and is not tmpfs/ramfs. On Kestrel,
  # TMPDIR on nodes without local disk can consume RAM, so spill jobs should
  # fall back to /scratch unless this test passes or DISCOVERY_SPILL_DIR is set.
  [[ -n "${TMPDIR:-}" ]] || return 1
  [[ -d "${TMPDIR}" ]] || return 1
  local fstype
  fstype="$(df -PT "${TMPDIR}" 2>/dev/null | awk 'NR==2 {print $2}')"
  [[ -n "${fstype}" ]] || return 1
  [[ ! "${fstype}" =~ ^(tmpfs|ramfs)$ ]]
}

kestrel_choose_spill_dir() {
  if [[ -n "${DISCOVERY_SPILL_DIR:-}" ]]; then
    printf '%s\n' "${DISCOVERY_SPILL_DIR}"
  elif kestrel_is_real_disk_tmpdir; then
    printf '%s\n' "${TMPDIR}/bsm_spill_probe"
  else
    printf '%s\n' "${KESTREL_SCRATCH_ROOT}/${SLURM_JOB_ID:-manual}/spill_probe"
  fi
}

kestrel_write_submit_command() {
  local output_file="$1"
  shift
  printf '%q ' "$@" > "${output_file}"
  printf '\n' >> "${output_file}"
}

kestrel_sacct_fields() {
  printf '%s\n' 'JobID,JobName%24,Partition,Account,State,ExitCode,Elapsed,Timelimit,AllocTRES,ReqTRES,MaxRSS,AveRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite,NodeList%40'
}

kestrel_monitor_slurm_job() {
  # Args:
  #   $1 job id
  #   $2 human-readable label, no spaces preferred
  #   $3 expected stdout file, or '-' if none
  #   $4 sacct output file
  local jobid="$1"
  local label="$2"
  local stdout_file="$3"
  local sacct_file="$4"
  local interval="${MONITOR_INTERVAL_SECONDS:-10}"
  local tail_lines="${MONITOR_TAIL_LINES:-8}"
  local last_state=""
  local last_reason=""
  local seen_running=0
  local queue_line state reason nodelist elapsed timelimit

  kestrel_status "job_label=${label} job_id=${jobid} status=SUBMITTED step=submitted monitor_interval_seconds=${interval}"

  while true; do
    queue_line="$(squeue -h -j "${jobid}" -o '%T|%R|%N|%M|%l' 2>/dev/null || true)"
    if [[ -z "${queue_line}" ]]; then
      kestrel_status "job_label=${label} job_id=${jobid} status=LEFT_QUEUE step=checking_final_accounting"
      break
    fi

    IFS='|' read -r state reason nodelist elapsed timelimit <<< "${queue_line}"

    if [[ "${state}" != "${last_state}" || "${reason}" != "${last_reason}" ]]; then
      case "${state}" in
        PENDING)
          kestrel_status "job_label=${label} job_id=${jobid} status=PENDING step=waiting_for_node reason=${reason} elapsed=${elapsed} timelimit=${timelimit}"
          ;;
        RUNNING)
          seen_running=1
          kestrel_status "job_label=${label} job_id=${jobid} status=RUNNING step=node_allocated nodes=${nodelist} elapsed=${elapsed} timelimit=${timelimit}"
          ;;
        CONFIGURING|COMPLETING)
          kestrel_status "job_label=${label} job_id=${jobid} status=${state} step=slurm_transition nodes=${nodelist} elapsed=${elapsed} timelimit=${timelimit}"
          ;;
        *)
          kestrel_status "job_label=${label} job_id=${jobid} status=${state} reason=${reason} nodes=${nodelist} elapsed=${elapsed} timelimit=${timelimit}"
          ;;
      esac
      last_state="${state}"
      last_reason="${reason}"
    elif [[ "${state}" == "PENDING" ]]; then
      kestrel_status "job_label=${label} job_id=${jobid} status=PENDING step=still_waiting reason=${reason} elapsed=${elapsed} timelimit=${timelimit}"
    elif [[ "${state}" == "RUNNING" ]]; then
      kestrel_status "job_label=${label} job_id=${jobid} status=RUNNING step=still_running nodes=${nodelist} elapsed=${elapsed} timelimit=${timelimit}"
    fi

    if [[ "${seen_running}" -eq 1 && "${stdout_file}" != "-" && -f "${stdout_file}" ]]; then
      kestrel_status "job_label=${label} job_id=${jobid} status=JOB_OUTPUT_TAIL file=${stdout_file} tail_lines=${tail_lines}"
      tail -n "${tail_lines}" "${stdout_file}" || true
    fi

    sleep "${interval}"
  done

  mkdir -p "$(dirname "${sacct_file}")"
  sacct -j "${jobid}" \
    --format="$(kestrel_sacct_fields)" \
    -P > "${sacct_file}" || true

  local final_state=""
  final_state="$(sacct -n -j "${jobid}" --format=State,ExitCode,Elapsed,NodeList -P 2>/dev/null | head -n 1 || true)"
  kestrel_status "job_label=${label} job_id=${jobid} status=FINAL sacct=${final_state:-unavailable} sacct_file=${sacct_file}"
}
