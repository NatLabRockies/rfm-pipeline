#!/usr/bin/env bash
set -euo pipefail

HPC_HOST="${HPC_HOST:-kl1.hpc.nrel.gov}"
HPC_USER="${HPC_USER:-${USER}}"
STUDY_ID="${STUDY_ID:-publication_full_dataset_distributed_20260519}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/${HPC_USER}/bsm/studies/${STUDY_ID}}"

SSH_TARGET="${HPC_USER}@${HPC_HOST}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/kestrel/status_publication_full_dataset_distributed.sh [options]

Show controller status and per-stage shard/reduce progress for the distributed full run.

Options:
  --host HOST          Kestrel host (default: kl1.hpc.nrel.gov)
  --user USER          Kestrel user (default: $USER)
  --study-id ID        Study identifier
  --study-root DIR     Study root on HPC
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

ssh -T "${SSH_TARGET}" "bash -lc '
set -euo pipefail
STUDY_ROOT=\"${STUDY_ROOT}\"
echo \"=== Distributed publication full run status (\$(date -u +%Y-%m-%dT%H:%M:%SZ)) ===\"
echo \"study_root=\${STUDY_ROOT}\"
echo

controller_file=\"\${STUDY_ROOT}/metadata/controller_job_id.txt\"
login_pid_file=\"\${STUDY_ROOT}/metadata/controller_login_pid.txt\"
login_log_file=\"\${STUDY_ROOT}/metadata/controller_login_log.txt\"
if [[ -f \"\${controller_file}\" ]]; then
  controller_job=\$(cat \"\${controller_file}\")
  controller_state=\$(sacct -X -n -j \"\${controller_job}\" --format=State 2>/dev/null | head -n1 | awk \"{print \\\$1}\")
  echo \"controller_job=\${controller_job} state=\${controller_state:-unknown}\"
else
  echo \"controller_job=missing\"
fi
if [[ -f \"\${login_pid_file}\" ]]; then
  login_pid=\$(cat \"\${login_pid_file}\")
  if ps -p \"\${login_pid}\" >/dev/null 2>&1; then
    login_state=\"RUNNING\"
  else
    login_state=\"NOT_RUNNING\"
  fi
  echo \"controller_login_pid=\${login_pid} state=\${login_state}\"
  if [[ -f \"\${login_log_file}\" ]]; then
    echo \"controller_login_log=\$(cat \"\${login_log_file}\")\"
  fi
fi
echo

stages=(output_conditioning empirical_null_screening interaction_discovery nonlinear_discovery sparse_selection final_manuscript_artifacts)
for stage in \"\${stages[@]}\"; do
  manifest=\"\${STUDY_ROOT}/hpc_scripts/\${stage}/manifest.jsonl\"
  shard_dir=\"\${STUDY_ROOT}/artifacts/hpc_shards_\${stage}\"
  merged_json=\"\${shard_dir}/_merged/\${stage}_merged.json\"
  expected=0
  succeeded=0
  failed=0
  if [[ -f \"\${manifest}\" ]]; then
    expected=\$(wc -l < \"\${manifest}\" | tr -d \" \")
  fi
  if [[ -d \"\${shard_dir}\" ]]; then
    succeeded=\$(find \"\${shard_dir}\" -mindepth 2 -maxdepth 2 -name _SUCCESS.json 2>/dev/null | wc -l | tr -d \" \")
    failed=\$(find \"\${shard_dir}\" -mindepth 3 -maxdepth 3 -name _FAILURE.json 2>/dev/null | wc -l | tr -d \" \")
  fi
  pending=\$(( expected - succeeded ))
  if [[ \${pending} -lt 0 ]]; then
    pending=0
  fi
  merged_status=\"missing\"
  if [[ -f \"\${merged_json}\" ]]; then
    merged_status=\"present\"
  fi
  echo \"[\${stage}] expected=\${expected} succeeded=\${succeeded} pending=\${pending} failures=\${failed} merged=\${merged_status}\"
done
echo

if [[ -f \"\${STUDY_ROOT}/metadata/stage_jobs.tsv\" ]]; then
  echo \"--- stage job registry ---\"
  cat \"\${STUDY_ROOT}/metadata/stage_jobs.tsv\"
fi

if [[ -f \"\${STUDY_ROOT}/metadata/RUN_COMPLETE\" ]]; then
  echo
  echo \"RUN_COMPLETE marker found.\"
elif [[ -f \"\${STUDY_ROOT}/metadata/RUN_FAILED_STAGE.txt\" ]]; then
  echo
  echo \"RUN_FAILED at stage: \$(cat \"\${STUDY_ROOT}/metadata/RUN_FAILED_STAGE.txt\")\"
fi
'"
