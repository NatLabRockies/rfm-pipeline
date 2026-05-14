#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=common_paths.sh
source "${REPO_ROOT}/scripts/kestrel/common_paths.sh"

HPC_HOST="${HPC_HOST:-kl1.hpc.nrel.gov}"
HPC_REPO_ROOT="${HPC_REPO_ROOT:-/projects/bsm/bsm-public-rf}"
HPC_ARTIFACTS_ROOT="${HPC_ARTIFACTS_ROOT:-__AUTO__}"
LOCAL_OUT_DIR="${LOCAL_OUT_DIR:-${REPO_ROOT}/artifacts/kestrel_collected_bundles}"
REMOTE_SNAPSHOT_ROOT="${REMOTE_SNAPSHOT_ROOT:-}"
PULLBACK_MODE="${PULLBACK_MODE:-reporting_bundle}"
KEEP_REMOTE=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/kestrel/pull_hpc_artifacts_bundle.sh [options]

Pull HPC run artifacts by:
  1) building a manifest and snapshot bundle on Kestrel
  2) downloading the zip locally
  3) writing a local analysis CSV/TXT from the embedded manifest

Options:
  --hpc-host HOST            SSH host/alias for Kestrel (default: $HPC_HOST)
  --hpc-repo-root DIR        Repo root on HPC (default: $HPC_REPO_ROOT)
  --hpc-artifacts-root DIR   Artifact root on HPC (default: auto; prefers /scratch/$USER/bsm/bsm-public-rf/artifacts)
  --local-out-dir DIR        Local destination for bundles (default: ./artifacts/kestrel_collected_bundles)
  --remote-snapshot-root DIR Remote snapshot root (default: auto)
  --pullback-mode MODE       manifest_only | reporting_bundle | full (default: reporting_bundle)
  --keep-remote              Keep remote snapshot dir + zip after download
  -h, --help                 Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hpc-host)
      HPC_HOST="${2:-}"
      shift 2
      ;;
    --hpc-repo-root)
      HPC_REPO_ROOT="${2:-}"
      shift 2
      ;;
    --hpc-artifacts-root)
      HPC_ARTIFACTS_ROOT="${2:-}"
      shift 2
      ;;
    --local-out-dir)
      LOCAL_OUT_DIR="${2:-}"
      shift 2
      ;;
    --remote-snapshot-root)
      REMOTE_SNAPSHOT_ROOT="${2:-}"
      shift 2
      ;;
    --pullback-mode)
      PULLBACK_MODE="${2:-}"
      shift 2
      ;;
    --keep-remote)
      KEEP_REMOTE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${PULLBACK_MODE}" in
  manifest_only|reporting_bundle|full)
    ;;
  *)
    echo "error: --pullback-mode must be one of: manifest_only, reporting_bundle, full" >&2
    exit 2
    ;;
esac

if [[ -z "${HPC_HOST}" || -z "${HPC_REPO_ROOT}" || -z "${LOCAL_OUT_DIR}" ]]; then
  echo "error: host/repo-root/local-out-dir cannot be empty" >&2
  exit 2
fi

mkdir -p "${LOCAL_OUT_DIR}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_SNAPSHOT_ARG="${REMOTE_SNAPSHOT_ROOT:-__AUTO__}"

echo "==> Building remote bundle on ${HPC_HOST} (mode=${PULLBACK_MODE})"
REMOTE_OUTPUT="$(ssh -T "${HPC_HOST}" "bash -s" -- \
  "${HPC_REPO_ROOT}" \
  "${TS}" \
  "${REMOTE_SNAPSHOT_ARG}" \
  "${HPC_ARTIFACTS_ROOT}" \
  "${PULLBACK_MODE}" <<'REMOTE_EOF'
set -euo pipefail

HPC_REPO_ROOT="$1"
TS="$2"
REMOTE_SNAPSHOT_ROOT="${3:-__AUTO__}"
HPC_ARTIFACTS_ROOT="${4:-__AUTO__}"
PULLBACK_MODE="${5:-reporting_bundle}"

if [[ "${REMOTE_SNAPSHOT_ROOT}" == "__AUTO__" ]]; then
  REMOTE_SNAPSHOT_ROOT=""
fi
case "${PULLBACK_MODE}" in
  manifest_only|reporting_bundle|full)
    ;;
  *)
    echo "error: invalid pullback mode: ${PULLBACK_MODE}" >&2
    exit 2
    ;;
esac

if [[ ! -d "${HPC_REPO_ROOT}" ]]; then
  echo "error: HPC repo root not found: ${HPC_REPO_ROOT}" >&2
  exit 2
fi

REMOTE_USER="$(id -un)"
if [[ "${HPC_ARTIFACTS_ROOT}" == "__AUTO__" ]]; then
  SCRATCH_ARTIFACTS_ROOT="/scratch/${REMOTE_USER}/bsm/bsm-public-rf/artifacts"
  if [[ -d "${SCRATCH_ARTIFACTS_ROOT}" || -L "${SCRATCH_ARTIFACTS_ROOT}" ]]; then
    HPC_ARTIFACTS_ROOT="${SCRATCH_ARTIFACTS_ROOT}"
  else
    HPC_ARTIFACTS_ROOT="${HPC_REPO_ROOT}/artifacts"
  fi
fi
if [[ "${HPC_ARTIFACTS_ROOT}" == "/home/${REMOTE_USER}/"* ]]; then
  echo "error: refusing home-directory artifacts root: ${HPC_ARTIFACTS_ROOT}" >&2
  exit 2
fi

if [[ -n "${REMOTE_SNAPSHOT_ROOT}" ]]; then
  SNAPSHOT_ROOT="${REMOTE_SNAPSHOT_ROOT}"
elif [[ "${HPC_REPO_ROOT}" == "/home/${REMOTE_USER}/"* ]]; then
  SNAPSHOT_ROOT="/scratch/${REMOTE_USER}/bsm/kestrel_hpc_snapshots"
else
  SNAPSHOT_ROOT="${HPC_REPO_ROOT}/artifacts/kestrel_hpc_snapshots"
fi
if [[ "${SNAPSHOT_ROOT}" == "/home/${REMOTE_USER}/"* ]]; then
  echo "error: refusing home-directory snapshot root: ${SNAPSHOT_ROOT}" >&2
  exit 2
fi

SNAPSHOT_DIR="${SNAPSHOT_ROOT}/${TS}"
BUNDLE_DIR="${SNAPSHOT_DIR}/bundle"
RUNS_DIR="${BUNDLE_DIR}/runs"
LOGS_DIR="${BUNDLE_DIR}/logs"
MANIFEST_DIR="${BUNDLE_DIR}/manifest"
ZIP_PATH="${SNAPSHOT_ROOT}/kestrel_hpc_snapshot_${TS}.zip"
LOGS_ROOT="/scratch/${REMOTE_USER}/bsm"
SUITE_ROOT="${HPC_ARTIFACTS_ROOT}/kestrel_cpu_scaling_suite"
MANIFEST_JSON="${MANIFEST_DIR}/hpc_run_manifest.json"
MANIFEST_CSV="${MANIFEST_DIR}/run_summary.csv"

mkdir -p "${RUNS_DIR}" "${LOGS_DIR}" "${MANIFEST_DIR}"

python3 "${HPC_REPO_ROOT}/tools/hpc_bundle_manifest.py" create-run-manifest \
  --hpc-repo-root "${HPC_REPO_ROOT}" \
  --artifacts-root "${HPC_ARTIFACTS_ROOT}" \
  --logs-root "${LOGS_ROOT}" \
  --suite-root "${SUITE_ROOT}" \
  --pullback-mode "${PULLBACK_MODE}" \
  --output-json "${MANIFEST_JSON}" \
  --output-csv "${MANIFEST_CSV}"

copy_tree() {
  local src="$1"
  local dst="$2"
  if [[ ! -d "${src}" ]]; then
    return 0
  fi
  mkdir -p "${dst}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${src}/" "${dst}/"
  else
    cp -a "${src}/." "${dst}/"
  fi
}

copy_file_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "${src}" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${dst}")"
  cp -f "${src}" "${dst}"
}

copy_latest_match() {
  local pattern="$1"
  local dst_dir="$2"
  local latest
  latest="$(ls -1t ${pattern} 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest}" && -f "${latest}" ]]; then
    mkdir -p "${dst_dir}"
    cp -f "${latest}" "${dst_dir}/"
  fi
}

copy_reporting_target_cpu() {
  local tier="$1"
  local run_dir="${HPC_ARTIFACTS_ROOT}/kestrel_cpu_scale_${tier}_run"
  local out_run="${RUNS_DIR}/cpu_${tier}"
  local out_logs="${LOGS_DIR}/cpu_${tier}"
  local log_dir="${LOGS_ROOT}/bsm_kestrel_cpu_scale_${tier}/logs"

  copy_tree "${run_dir}/hpc_scripts" "${out_run}/hpc_scripts"
  copy_tree "${run_dir}/hpc_shards/_merged" "${out_run}/hpc_shards/_merged"
  copy_file_if_exists "${SUITE_ROOT}/cpu_nodes_${tier}/hpc_scripts/manifest.jsonl" \
    "${out_run}/hpc_scripts/suite_manifest.jsonl"
  copy_latest_match "${log_dir}/bsm_interaction_discovery_*.out" "${out_logs}"
  copy_latest_match "${log_dir}/bsm_reduce_interaction_discovery_*.out" "${out_logs}"
}

copy_reporting_target_gpu() {
  local run_dir="${HPC_ARTIFACTS_ROOT}/kestrel_gpu_h100_run"
  local out_run="${RUNS_DIR}/gpu_h100"
  local out_logs="${LOGS_DIR}/gpu_h100"
  local log_dir="${LOGS_ROOT}/bsm_kestrel_gpu_h100/logs"

  copy_tree "${run_dir}/hpc_scripts" "${out_run}/hpc_scripts"
  copy_tree "${run_dir}/hpc_shards/_merged" "${out_run}/hpc_shards/_merged"
  copy_latest_match "${log_dir}/bsm_gpu_interaction_discovery_*.out" "${out_logs}"
  copy_latest_match "${log_dir}/bsm_interaction_discovery_*.out" "${out_logs}"
  copy_latest_match "${log_dir}/bsm_reduce_interaction_discovery_*.out" "${out_logs}"
}

if [[ "${PULLBACK_MODE}" == "full" ]]; then
  if [[ -d "${SUITE_ROOT}" || -L "${SUITE_ROOT}" ]]; then
    copy_tree "${SUITE_ROOT}" "${RUNS_DIR}/cpu_scaling_suite"
  else
    copy_tree "${HPC_REPO_ROOT}/artifacts/kestrel_cpu_scaling_suite" "${RUNS_DIR}/cpu_scaling_suite"
  fi
  for tier in 2 10 1000; do
    run_dir="${HPC_ARTIFACTS_ROOT}/kestrel_cpu_scale_${tier}_run"
    copy_tree "${run_dir}/hpc_scripts" "${RUNS_DIR}/cpu_${tier}/hpc_scripts"
    copy_tree "${run_dir}/hpc_shards" "${RUNS_DIR}/cpu_${tier}/hpc_shards"
    copy_tree "${LOGS_ROOT}/bsm_kestrel_cpu_scale_${tier}/logs" "${LOGS_DIR}/cpu_${tier}"
  done
  gpu_run_dir="${HPC_ARTIFACTS_ROOT}/kestrel_gpu_h100_run"
  copy_tree "${gpu_run_dir}/hpc_scripts" "${RUNS_DIR}/gpu_h100/hpc_scripts"
  copy_tree "${gpu_run_dir}/hpc_shards" "${RUNS_DIR}/gpu_h100/hpc_shards"
  copy_tree "${LOGS_ROOT}/bsm_kestrel_gpu_h100/logs" "${LOGS_DIR}/gpu_h100"
elif [[ "${PULLBACK_MODE}" == "reporting_bundle" ]]; then
  copy_tree "${SUITE_ROOT}" "${RUNS_DIR}/cpu_scaling_suite"
  for tier in 2 10 1000; do
    copy_reporting_target_cpu "${tier}"
  done
  copy_reporting_target_gpu
fi

python3 - "${BUNDLE_DIR}" "${ZIP_PATH}" <<'PY'
import os
import sys
import zipfile
from pathlib import Path

bundle_dir = Path(sys.argv[1]).resolve()
zip_path = Path(sys.argv[2]).resolve()
zip_path.parent.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(str(bundle_dir)):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            arcname = os.path.relpath(fpath, str(bundle_dir))
            zf.write(fpath, arcname=arcname)
PY

echo "REMOTE_ZIP_PATH=${ZIP_PATH}"
echo "REMOTE_SNAPSHOT_DIR=${SNAPSHOT_DIR}"
echo "REMOTE_MANIFEST_JSON=${MANIFEST_JSON}"
echo "REMOTE_MANIFEST_CSV=${MANIFEST_CSV}"
REMOTE_EOF
)"

REMOTE_ZIP_PATH="$(printf '%s\n' "${REMOTE_OUTPUT}" | awk -F= '/^REMOTE_ZIP_PATH=/{print $2}' | tail -n 1)"
REMOTE_SNAPSHOT_DIR="$(printf '%s\n' "${REMOTE_OUTPUT}" | awk -F= '/^REMOTE_SNAPSHOT_DIR=/{print $2}' | tail -n 1)"

if [[ -z "${REMOTE_ZIP_PATH}" ]]; then
  echo "error: failed to parse REMOTE_ZIP_PATH from ssh output" >&2
  printf '%s\n' "${REMOTE_OUTPUT}" >&2
  exit 1
fi

echo "==> Downloading ${REMOTE_ZIP_PATH}"
scp "${HPC_HOST}:${REMOTE_ZIP_PATH}" "${LOCAL_OUT_DIR}/"

LOCAL_ZIP_PATH="${LOCAL_OUT_DIR}/$(basename "${REMOTE_ZIP_PATH}")"
LOCAL_ANALYSIS_CSV="${LOCAL_ZIP_PATH%.zip}_analysis.csv"
LOCAL_ANALYSIS_TXT="${LOCAL_ZIP_PATH%.zip}_analysis.txt"

if command -v pixi >/dev/null 2>&1; then
  PY_RUNNER=(pixi run python)
else
  PY_RUNNER=(python3)
fi

"${PY_RUNNER[@]}" "${REPO_ROOT}/tools/hpc_bundle_manifest.py" analyze-zip \
  --zip "${LOCAL_ZIP_PATH}" \
  --out-csv "${LOCAL_ANALYSIS_CSV}" \
  --out-txt "${LOCAL_ANALYSIS_TXT}"

if [[ "${KEEP_REMOTE}" -eq 0 ]]; then
  echo "==> Cleaning remote temporary bundle"
  ssh -T "${HPC_HOST}" "bash -s" -- "${REMOTE_ZIP_PATH}" "${REMOTE_SNAPSHOT_DIR}" <<'REMOTE_CLEAN_EOF'
set -euo pipefail
zip_path="$1"
snapshot_dir="$2"
[[ -n "${zip_path}" ]] && rm -f "${zip_path}" || true
[[ -n "${snapshot_dir}" ]] && rm -rf "${snapshot_dir}" || true
REMOTE_CLEAN_EOF
fi

echo "==> Done"
echo "Local bundle: ${LOCAL_ZIP_PATH}"
echo "Local analysis CSV: ${LOCAL_ANALYSIS_CSV}"
echo "Local analysis TXT: ${LOCAL_ANALYSIS_TXT}"
