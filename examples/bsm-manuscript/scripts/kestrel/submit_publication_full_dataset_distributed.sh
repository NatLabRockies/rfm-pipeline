#!/usr/bin/env bash
set -euo pipefail

echo "RETIRED: use the content-addressed G11 campaign package; this legacy workflow is non-executable." >&2
exit 64

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

STUDY_ID="publication_full_dataset_distributed_20260519"  # Original publication run date; override with --study-id for your run
STUDY_ROOT=""
DATASET_PATH="${DATASET_PATH:-/scratch/${USER}/bsm/bsm-public-rf/artifacts/preprocessed_real_data_30k}"
CONTROLLER_WALLTIME="48:00:00"
PARTITION="shared"
ACCOUNT="bsm"
LAUNCH_MODE="login"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/kestrel/submit_publication_full_dataset_distributed.sh [options]

Submit the full end-to-end distributed publication run controller job on Kestrel.

Options:
  --study-id ID             Study identifier (default: publication_full_dataset_distributed_20260519)
  --study-root DIR          Remote study root (default: /scratch/$USER/bsm/studies/<study-id>)
  --dataset-path DIR        Full dataset path on HPC
  --walltime HH:MM:SS       Controller walltime (default: 48:00:00)
  --partition NAME          Controller partition (default: shared)
  --account NAME            SLURM account (default: bsm)
  --launch-mode MODE        login | sbatch (default: login)
  -h, --help                Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --study-id)
      STUDY_ID="${2:-}"
      shift 2
      ;;
    --study-root)
      STUDY_ROOT="${2:-}"
      shift 2
      ;;
    --dataset-path)
      DATASET_PATH="${2:-}"
      shift 2
      ;;
    --walltime)
      CONTROLLER_WALLTIME="${2:-}"
      shift 2
      ;;
    --partition)
      PARTITION="${2:-}"
      shift 2
      ;;
    --account)
      ACCOUNT="${2:-}"
      shift 2
      ;;
    --launch-mode)
      LAUNCH_MODE="${2:-}"
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

if [[ -z "${STUDY_ROOT}" ]]; then
  STUDY_ROOT="/scratch/${USER}/bsm/studies/${STUDY_ID}"
fi

if [[ "${LAUNCH_MODE}" != "login" && "${LAUNCH_MODE}" != "sbatch" ]]; then
  echo "error: --launch-mode must be login or sbatch" >&2
  exit 2
fi

CONTROLLER_SCRIPT="${REPO_ROOT}/scripts/kestrel/controller_publication_full_dataset_distributed.sh"
if [[ ! -f "${CONTROLLER_SCRIPT}" ]]; then
  echo "error: missing controller script: ${CONTROLLER_SCRIPT}" >&2
  exit 2
fi

mkdir -p "${STUDY_ROOT}/logs" "${STUDY_ROOT}/metadata" "${STUDY_ROOT}/hpc_scripts"

export PIXI_HOME="${PIXI_HOME:-/projects/bsm/.pixi}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-/projects/bsm/.cache/pixi}"

cd "${REPO_ROOT}"
pixi install --locked

LAUNCH_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat > "${STUDY_ROOT}/metadata/launch_context.env" <<EOF
STUDY_ID=${STUDY_ID}
STUDY_ROOT=${STUDY_ROOT}
DATASET_PATH=${DATASET_PATH}
REPO_ROOT=${REPO_ROOT}
LAUNCH_TS=${LAUNCH_TS}
EOF

if [[ -d "${REPO_ROOT}/.git" ]]; then
  git --no-pager rev-parse HEAD > "${STUDY_ROOT}/metadata/repo_commit.txt" || true
  git --no-pager status --short > "${STUDY_ROOT}/metadata/repo_status.txt" || true
  git --no-pager diff > "${STUDY_ROOT}/metadata/repo_diff.patch" || true
fi

if [[ "${LAUNCH_MODE}" == "sbatch" ]]; then
  CONTROLLER_JOB_ID="$(
    sbatch --parsable \
      --account="${ACCOUNT}" \
      --partition="${PARTITION}" \
      --time="${CONTROLLER_WALLTIME}" \
      --mem=8G \
      --cpus-per-task=1 \
      --job-name=bsm_pub_full_dist_ctl \
      --output="${STUDY_ROOT}/logs/controller_%j.out" \
      --error="${STUDY_ROOT}/logs/controller_%j.err" \
      --export=ALL,REPO_ROOT="${REPO_ROOT}",STUDY_ID="${STUDY_ID}",STUDY_ROOT="${STUDY_ROOT}",DATASET_PATH="${DATASET_PATH}" \
      "${CONTROLLER_SCRIPT}"
  )"
  echo "${CONTROLLER_JOB_ID}" > "${STUDY_ROOT}/metadata/controller_job_id.txt"
  rm -f "${STUDY_ROOT}/metadata/controller_login_pid.txt" "${STUDY_ROOT}/metadata/controller_login_log.txt"
  echo "Submitted controller job: ${CONTROLLER_JOB_ID}"
  echo "Study root: ${STUDY_ROOT}"
  echo "Monitor:"
  echo "  squeue -j ${CONTROLLER_JOB_ID}"
  echo "  tail -f ${STUDY_ROOT}/logs/controller_${CONTROLLER_JOB_ID}.out"
else
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  login_log="${STUDY_ROOT}/logs/controller_login_${ts}.out"
  REPO_ROOT="${REPO_ROOT}" \
    STUDY_ID="${STUDY_ID}" \
    STUDY_ROOT="${STUDY_ROOT}" \
    DATASET_PATH="${DATASET_PATH}" \
    nohup bash "${CONTROLLER_SCRIPT}" > "${login_log}" 2>&1 < /dev/null &
  login_pid=$!
  echo "${login_pid}" > "${STUDY_ROOT}/metadata/controller_login_pid.txt"
  echo "${login_log}" > "${STUDY_ROOT}/metadata/controller_login_log.txt"
  rm -f "${STUDY_ROOT}/metadata/controller_job_id.txt"
  echo "Started controller on login host: pid=${login_pid}"
  echo "Study root: ${STUDY_ROOT}"
  echo "Monitor:"
  echo "  ps -p ${login_pid}"
  echo "  tail -f ${login_log}"
fi
