#!/usr/bin/env bash
set -euo pipefail

cd "/home/dhetting/src/bsm-public-rf/scripts"
source "/home/dhetting/src/bsm-public-rf/scripts/hpc_kestrel_config.sh"
export DISCOVERY_OUT_DIR="/home/dhetting/src/bsm-public-rf/scripts/artifacts/hpc_discovery"
export PATH="/projects/bsm/.pixi/bin:${PATH}"

if ! command -v pixi >/dev/null 2>&1; then
  echo "ERROR: pixi not found in job. Run ./install_pixi_project_env.sh before submitting Python probes." >&2
  exit 2
fi

bash "/home/dhetting/src/bsm-public-rf/scripts/discovery_chunk_spill_probe.sh" "/home/dhetting/src/bsm-public-rf/scripts/artifacts/hpc_discovery"
bash "/home/dhetting/src/bsm-public-rf/scripts/discovery_parquet_bench.sh" "/home/dhetting/src/bsm-public-rf/scripts/artifacts/hpc_discovery"
