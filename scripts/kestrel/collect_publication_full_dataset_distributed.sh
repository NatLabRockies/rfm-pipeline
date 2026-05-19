#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

HPC_HOST="${HPC_HOST:-kl1.hpc.nrel.gov}"
HPC_USER="${HPC_USER:-${USER}}"
STUDY_ID="${STUDY_ID:-publication_full_dataset_distributed_20260519}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/${HPC_USER}/bsm/studies/${STUDY_ID}}"
LOCAL_OUT_DIR="${LOCAL_OUT_DIR:-${REPO_ROOT}/artifacts/publication_full_dataset_distributed_results}"
MODE="${MODE:-reporting}"

SSH_TARGET="${HPC_USER}@${HPC_HOST}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/kestrel/collect_publication_full_dataset_distributed.sh [options]

Collect distributed-run artifacts from Kestrel to local machine.

Options:
  --host HOST          Kestrel host (default: kl1.hpc.nrel.gov)
  --user USER          Kestrel user (default: $USER)
  --study-id ID        Study identifier
  --study-root DIR     Study root on HPC
  --local-out-dir DIR  Local destination root
  --mode MODE          reporting | full (default: reporting)
  -h, --help           Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HPC_HOST="${2:-}"
      shift 2
      ;;
    --user)
      HPC_USER="${2:-}"
      shift 2
      ;;
    --study-id)
      STUDY_ID="${2:-}"
      shift 2
      ;;
    --study-root)
      STUDY_ROOT="${2:-}"
      shift 2
      ;;
    --local-out-dir)
      LOCAL_OUT_DIR="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
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

if [[ "${MODE}" != "reporting" && "${MODE}" != "full" ]]; then
  echo "error: --mode must be reporting or full" >&2
  exit 2
fi

SSH_TARGET="${HPC_USER}@${HPC_HOST}"
DEST_ROOT="${LOCAL_OUT_DIR}/${STUDY_ID}"
mkdir -p "${DEST_ROOT}"

if [[ "${MODE}" == "full" ]]; then
  echo "==> Collecting full study tree from ${SSH_TARGET}:${STUDY_ROOT}"
  rsync -az "${SSH_TARGET}:${STUDY_ROOT}/" "${DEST_ROOT}/"
else
  echo "==> Collecting reporting bundle from ${SSH_TARGET}:${STUDY_ROOT}"
  for rel in \
    metadata \
    logs \
    hpc_scripts \
    artifacts/output_conditioning \
    artifacts/empirical_null_screen \
    artifacts/interaction_discovery \
    artifacts/nonlinear_discovery \
    artifacts/sparse_selection \
    artifacts/final_manuscript_artifacts \
    artifacts/runtime_diagnostics; do
    rsync -az "${SSH_TARGET}:${STUDY_ROOT}/${rel}/" "${DEST_ROOT}/${rel}/" || true
  done
  for stage in \
    output_conditioning \
    empirical_null_screening \
    interaction_discovery \
    nonlinear_discovery \
    sparse_selection \
    final_manuscript_artifacts; do
    rsync -az \
      "${SSH_TARGET}:${STUDY_ROOT}/artifacts/hpc_shards_${stage}/_merged/" \
      "${DEST_ROOT}/artifacts/hpc_shards_${stage}/_merged/" || true
  done
fi

echo "Collection complete: ${DEST_ROOT}"
