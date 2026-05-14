#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./mkdir_hpc_dirs.sh [host]

Creates Kestrel directory trees needed by CPU scaling runs:
  - <REMOTE_ARTIFACTS_ROOT>/kestrel_cpu_scale_{2,10,1000}_run (+ stage subdirs)
  - <REMOTE_SUITE_ROOT>/{diagnostic,cpu_nodes_*}/hpc_scripts
  - /scratch/$USER/bsm/bsm_kestrel_cpu_scale_{2,10,1000}/logs

Default host: kl1.hpc.nrel.gov

Environment overrides:
  REMOTE_REPO_ROOT=/projects/bsm/bsm-public-rf
  REMOTE_ARTIFACTS_ROOT=/scratch/$USER/bsm/bsm-public-rf/artifacts
  REMOTE_SUITE_ROOT=/projects/bsm/bsm-public-rf/artifacts/kestrel_cpu_scaling_suite
EOF
  exit 0
fi

HOST="${1:-kl1.hpc.nrel.gov}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/projects/bsm/bsm-public-rf}"
REMOTE_ARTIFACTS_ROOT="${REMOTE_ARTIFACTS_ROOT:-/scratch/${USER}/bsm/bsm-public-rf/artifacts}"
REMOTE_SUITE_ROOT="${REMOTE_SUITE_ROOT:-${REMOTE_REPO_ROOT}/artifacts/kestrel_cpu_scaling_suite}"

ssh "$HOST" "bash -s" -- "${REMOTE_ARTIFACTS_ROOT}" "${REMOTE_SUITE_ROOT}" <<'EOF'
set -euo pipefail
REMOTE_ARTIFACTS_ROOT="$1"
REMOTE_SUITE_ROOT="$2"

for n in 2 10 1000; do
  run_dir="${REMOTE_ARTIFACTS_ROOT}/kestrel_cpu_scale_${n}_run"
  mkdir -p \
    "$run_dir" \
    "$run_dir/output_conditioning" \
    "$run_dir/empirical_null_screen" \
    "$run_dir/interaction_discovery" \
    "$run_dir/nonlinear_discovery" \
    "$run_dir/sparse_selection" \
    "$run_dir/final_artifacts"

  mkdir -p \
    "${REMOTE_SUITE_ROOT}/cpu_nodes_${n}/hpc_scripts" \
    /scratch/${USER}/bsm/bsm_kestrel_cpu_scale_${n}/logs
done

mkdir -p "${REMOTE_SUITE_ROOT}/diagnostic/hpc_scripts"
EOF

echo "Done: created Kestrel CPU scaling directories on ${HOST}"
