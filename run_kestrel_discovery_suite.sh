#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_kestrel_config.sh
source "${SCRIPT_DIR}/hpc_kestrel_config.sh"

OUT_DIR="${1:-${DISCOVERY_OUT_DIR}}"
OUT_DIR="$(kestrel_realpath "${OUT_DIR}")"
mkdir -p "${OUT_DIR}"

kestrel_status "script=run_kestrel_discovery_suite.sh step=start_discovery_suite out_dir=${OUT_DIR} account=${NLR_ACCOUNT} cpu_partition=${KESTREL_CPU_PARTITION} gpu_partition=${KESTREL_GPU_PARTITION}"

kestrel_status "script=run_kestrel_discovery_suite.sh step=collect_login_node_control_plane_details"
bash "${SCRIPT_DIR}/discovery_collect_env.sh" "${OUT_DIR}"

kestrel_status "script=run_kestrel_discovery_suite.sh step=submit_cpu_debug_slurm_topology_probe"
bash "${SCRIPT_DIR}/discovery_collect_slurm_job.sh" "${OUT_DIR}"

if command -v pixi >/dev/null 2>&1; then
  kestrel_status "script=run_kestrel_discovery_suite.sh step=submit_cpu_debug_python_pixi_spill_parquet_probes"
  bash "${SCRIPT_DIR}/discovery_submit_python_probes.sh" "${OUT_DIR}"
else
  kestrel_status "script=run_kestrel_discovery_suite.sh status=SKIPPED step=python_pixi_probes reason=pixi_not_on_path hint=run_install_pixi_project_env_first"
  echo ">>> Skipping Python/Pixi probes because pixi is not on PATH. Run ./install_pixi_project_env.sh first."
fi

if kestrel_have_partition "${KESTREL_GPU_PARTITION}"; then
  kestrel_status "script=run_kestrel_discovery_suite.sh step=submit_optional_short_gpu_probe partition=${KESTREL_GPU_PARTITION}"
  bash "${SCRIPT_DIR}/discovery_gpu_probe.sh" "${OUT_DIR}"
else
  kestrel_status "script=run_kestrel_discovery_suite.sh status=SKIPPED step=gpu_probe reason=partition_not_visible partition=${KESTREL_GPU_PARTITION}"
  echo ">>> Skipping GPU probe: ${KESTREL_GPU_PARTITION} not visible from this session."
fi

kestrel_status "script=run_kestrel_discovery_suite.sh step=done_discovery_suite out_dir=${OUT_DIR}"
echo "Wrote discovery artifacts to ${OUT_DIR}"
