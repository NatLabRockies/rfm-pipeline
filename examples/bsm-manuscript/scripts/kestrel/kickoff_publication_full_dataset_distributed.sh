#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# CONFIGURE: set HPC_HOST env var or edit this line to your HPC login node
HPC_HOST="${HPC_HOST:-your.hpc.login.node}"
HPC_USER="${HPC_USER:-${USER}}"
STUDY_ID="${STUDY_ID:-publication_full_dataset_distributed_20260519}"
REMOTE_STUDY_ROOT="${REMOTE_STUDY_ROOT:-/scratch/${HPC_USER}/bsm/studies/${STUDY_ID}}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/home/${HPC_USER}/src/bsm-public-rf}"
DATASET_PATH="${DATASET_PATH:-/scratch/${USER}/bsm/bsm-public-rf/artifacts/preprocessed_real_data_30k}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/kestrel/kickoff_publication_full_dataset_distributed.sh [options]

Git-based kickoff (no rsync): require local clean branch, push, pull on HPC, then submit.

Options:
  --host HOST               HPC host (default: your.hpc.login.node)
  --user USER               Kestrel username (default: $USER)
  --study-id ID             Study identifier
  --remote-study-root DIR   Root directory for this study on HPC
  --remote-repo-root DIR    Git repo root on HPC (default: /home/<user>/src/bsm-public-rf)
  --dataset-path DIR        Full dataset path on HPC
  -h, --help                Show this help
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
    --remote-study-root)
      REMOTE_STUDY_ROOT="${2:-}"
      shift 2
      ;;
    --remote-repo-root)
      REMOTE_REPO_ROOT="${2:-}"
      shift 2
      ;;
    --dataset-path)
      DATASET_PATH="${2:-}"
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

SSH_TARGET="${HPC_USER}@${HPC_HOST}"
if [[ -z "${REMOTE_STUDY_ROOT}" ]]; then
  REMOTE_STUDY_ROOT="/scratch/${HPC_USER}/bsm/studies/${STUDY_ID}"
fi

LOCAL_BRANCH="$(cd "${REPO_ROOT}" && git --no-pager branch --show-current)"
LOCAL_COMMIT="$(cd "${REPO_ROOT}" && git --no-pager rev-parse HEAD)"
if [[ -z "${LOCAL_BRANCH}" ]]; then
  echo "error: unable to resolve local branch" >&2
  exit 2
fi

if [[ -n "$(cd "${REPO_ROOT}" && git --no-pager status --porcelain)" ]]; then
  echo "error: local repo is not clean. Commit/test/push first, then rerun kickoff." >&2
  exit 2
fi

echo "==> Ensuring remote study directory exists"
ssh -T "${SSH_TARGET}" \
  "mkdir -p '${REMOTE_STUDY_ROOT}' '${REMOTE_STUDY_ROOT}/logs' '${REMOTE_STUDY_ROOT}/metadata'"

echo "==> Pulling latest git branch on HPC"
ssh -T "${SSH_TARGET}" "bash -lc 'set -euo pipefail; \
  cd \"${REMOTE_REPO_ROOT}\" && \
  git --no-pager fetch --all --prune && \
  git --no-pager checkout \"${LOCAL_BRANCH}\" && \
  git --no-pager pull --ff-only origin \"${LOCAL_BRANCH}\"'"

REMOTE_COMMIT="$(
  ssh -T "${SSH_TARGET}" "bash -lc 'cd \"${REMOTE_REPO_ROOT}\" && git --no-pager rev-parse HEAD'"
)"
if [[ "${REMOTE_COMMIT}" != "${LOCAL_COMMIT}" ]]; then
  echo "error: remote commit (${REMOTE_COMMIT}) does not match local commit (${LOCAL_COMMIT})" >&2
  exit 2
fi

echo "==> Preparing remote study directory"
ssh -T "${SSH_TARGET}" \
  "bash -lc 'cat > \"${REMOTE_STUDY_ROOT}/metadata/git_sync.env\" <<EOF
study_id=${STUDY_ID}
branch=${LOCAL_BRANCH}
commit=${LOCAL_COMMIT}
remote_repo_root=${REMOTE_REPO_ROOT}
synced_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF'"

echo "==> Submitting distributed full-dataset controller job"
ssh -T "${SSH_TARGET}" "bash -lc 'cd \"${REMOTE_REPO_ROOT}\" && \
  export PIXI_HOME=/projects/bsm/.pixi PIXI_CACHE_DIR=/projects/bsm/.cache/pixi && \
  bash scripts/kestrel/submit_publication_full_dataset_distributed.sh \
    --study-id \"${STUDY_ID}\" \
    --study-root \"${REMOTE_STUDY_ROOT}\" \
    --launch-mode login \
    --dataset-path \"${DATASET_PATH}\"'"

echo
echo "Kickoff complete."
echo "Git sync:"
echo "  branch=${LOCAL_BRANCH}"
echo "  commit=${LOCAL_COMMIT}"
echo "Tracking:"
echo "  bash scripts/kestrel/status_publication_full_dataset_distributed.sh --host ${HPC_HOST} --user ${HPC_USER} --study-root ${REMOTE_STUDY_ROOT}"
echo "Collection:"
echo "  bash scripts/kestrel/collect_publication_full_dataset_distributed.sh --host ${HPC_HOST} --user ${HPC_USER} --study-root ${REMOTE_STUDY_ROOT}"
