#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scp.sh [host]

Copies all interaction_discovery prerequisite artifacts to Kestrel CPU scaling run dirs:
  - <REMOTE_ARTIFACTS_ROOT>/kestrel_cpu_scale_2_run
  - <REMOTE_ARTIFACTS_ROOT>/kestrel_cpu_scale_10_run
  - <REMOTE_ARTIFACTS_ROOT>/kestrel_cpu_scale_1000_run

Default host: kl1.hpc.nrel.gov

Environment overrides:
  REMOTE_REPO_ROOT=/projects/bsm/bsm-public-rf
  REMOTE_ARTIFACTS_ROOT=/scratch/$USER/bsm/bsm-public-rf/artifacts
EOF
  exit 0
fi

HOST="${1:-kl1.hpc.nrel.gov}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/projects/bsm/bsm-public-rf}"
REMOTE_ARTIFACTS_ROOT="${REMOTE_ARTIFACTS_ROOT:-/scratch/${USER}/bsm/bsm-public-rf/artifacts}"
REMOTE_ARTIFACTS_ROOT="${REMOTE_ARTIFACTS_ROOT%/}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC_PCA="${LOCAL_ROOT}/artifacts/validation_300_sample_no_caps/output_conditioning/pca_scores.csv"
SRC_RETAINED="${LOCAL_ROOT}/artifacts/validation_300_sample_no_caps/empirical_null_screen/retained_terms.csv"
SRC_X="${LOCAL_ROOT}/artifacts/test_dataset_300/X.parquet"
SRC_HOLDOUT="${LOCAL_ROOT}/artifacts/test_dataset_300/holdout_assignments.parquet"
SRC_CATALOG="${LOCAL_ROOT}/artifacts/actual_input_feature_catalog.parquet"

for f in "$SRC_PCA" "$SRC_RETAINED" "$SRC_X" "$SRC_HOLDOUT" "$SRC_CATALOG"; do
  [[ -f "$f" ]] || { echo "Missing local source file: $f" >&2; exit 1; }
done

for n in 2 10 1000; do
  RUN_DIR="${REMOTE_ARTIFACTS_ROOT}/kestrel_cpu_scale_${n}_run"
  echo "Copying prerequisites to ${HOST}:${RUN_DIR}"
  ssh "$HOST" "mkdir -p '${RUN_DIR}/output_conditioning' '${RUN_DIR}/empirical_null_screen'"
  scp "$SRC_PCA" "$HOST:${RUN_DIR}/output_conditioning/pca_scores.csv"
  scp "$SRC_RETAINED" "$HOST:${RUN_DIR}/empirical_null_screen/retained_terms.csv"
  scp "$SRC_X" "$HOST:${RUN_DIR}/X.parquet"
  scp "$SRC_HOLDOUT" "$HOST:${RUN_DIR}/holdout_assignments.parquet"
  scp "$SRC_CATALOG" "$HOST:${RUN_DIR}/actual_input_feature_catalog.parquet"
done

echo "Done."
