#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_kestrel_config.sh
source "${SCRIPT_DIR}/hpc_kestrel_config.sh"

OUT_DIR="${1:-${DISCOVERY_OUT_DIR}}"
OUT_DIR="$(kestrel_realpath "${OUT_DIR}")"
mkdir -p "${OUT_DIR}"

echo ">>> Collecting login-node/control-plane environment details"
bash "${SCRIPT_DIR}/discovery_collect_env.sh" "${OUT_DIR}"

echo ">>> Submitting CPU debug Slurm topology probe"
bash "${SCRIPT_DIR}/discovery_collect_slurm_job.sh" "${OUT_DIR}"

if command -v pixi >/dev/null 2>&1; then
  echo ">>> Submitting CPU debug Python/Pixi spill and Parquet probes"
  bash "${SCRIPT_DIR}/discovery_submit_python_probes.sh" "${OUT_DIR}"
else
  echo ">>> Skipping Python/Pixi probes because pixi is not on PATH. Run ./install_pixi_project_env.sh first."
fi

if kestrel_have_partition "${KESTREL_GPU_PARTITION}"; then
  echo ">>> Submitting optional short GPU probe on ${KESTREL_GPU_PARTITION}"
  bash "${SCRIPT_DIR}/discovery_gpu_probe.sh" "${OUT_DIR}"
else
  echo ">>> Skipping GPU probe: ${KESTREL_GPU_PARTITION} not visible from this session."
fi

echo "Wrote discovery artifacts to ${OUT_DIR}"
