#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_kestrel_config.sh
source "${SCRIPT_DIR}/hpc_kestrel_config.sh"

OUT_DIR="${1:-${DISCOVERY_OUT_DIR}}"
OUT_DIR="$(kestrel_realpath "${OUT_DIR}")"
mkdir -p "${OUT_DIR}" "${PIXI_HOME}" "${PIXI_CACHE_DIR}" "${XDG_CACHE_HOME}" "${CONDA_PKGS_DIRS}"

LOG="${OUT_DIR}/pixi_install.log"
kestrel_status "script=install_pixi_project_env.sh step=start_pixi_install out_dir=${OUT_DIR} project_root=${KESTREL_PROJECT_ROOT} pixi_home=${PIXI_HOME}"
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "user=$(whoami)"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "NLR_ACCOUNT=${NLR_ACCOUNT}"
  echo "KESTREL_PROJECT_ROOT=${KESTREL_PROJECT_ROOT}"
  echo "PIXI_HOME=${PIXI_HOME}"
  echo "PIXI_CACHE_DIR=${PIXI_CACHE_DIR}"
  echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
  echo "CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS}"
  echo "PATH=${PATH}"
} > "${LOG}"

kestrel_status "script=install_pixi_project_env.sh step=check_project_root_writable path=${KESTREL_PROJECT_ROOT}"
if ! [[ -w "${KESTREL_PROJECT_ROOT}" ]]; then
  echo "ERROR: ${KESTREL_PROJECT_ROOT} is not writable. Check allocation/group permissions." | tee -a "${LOG}" >&2
  exit 2
fi

kestrel_status "script=install_pixi_project_env.sh step=check_existing_pixi"
if ! command -v pixi >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: pixi is not on PATH and curl is unavailable; install pixi manually into ${PIXI_HOME}." | tee -a "${LOG}" >&2
    exit 3
  fi
  kestrel_status "script=install_pixi_project_env.sh step=install_pixi_binary pixi_home=${PIXI_HOME}"
  echo ">>> Installing pixi into ${PIXI_HOME}" | tee -a "${LOG}"
  curl -fsSL https://pixi.sh/install.sh | PIXI_HOME="${PIXI_HOME}" bash 2>&1 | tee -a "${LOG}"
else
  kestrel_status "script=install_pixi_project_env.sh step=pixi_already_available path=$(command -v pixi)"
  echo ">>> pixi already available at $(command -v pixi)" | tee -a "${LOG}"
fi

export PATH="${PIXI_HOME}/bin:${PATH}"
if ! command -v pixi >/dev/null 2>&1; then
  echo "ERROR: pixi still not found after install. Expected ${PIXI_HOME}/bin/pixi." | tee -a "${LOG}" >&2
  exit 4
fi

kestrel_status "script=install_pixi_project_env.sh step=record_pixi_version"
pixi --version 2>&1 | tee -a "${LOG}"

if [[ -f pixi.toml ]]; then
  if [[ -f pixi.lock ]]; then
    kestrel_status "script=install_pixi_project_env.sh step=pixi_install_locked"
    echo ">>> Running pixi install --locked" | tee -a "${LOG}"
    pixi install --locked 2>&1 | tee -a "${LOG}"
  else
    kestrel_status "script=install_pixi_project_env.sh step=pixi_install_unlocked"
    echo ">>> Running pixi install" | tee -a "${LOG}"
    pixi install 2>&1 | tee -a "${LOG}"
  fi
else
  kestrel_status "script=install_pixi_project_env.sh step=no_pixi_toml_binary_only pwd=$(pwd)"
  echo ">>> No pixi.toml found in $(pwd); installed pixi binary/cache only." | tee -a "${LOG}"
fi

kestrel_status "script=install_pixi_project_env.sh step=write_env_file"
cat > "${OUT_DIR}/pixi_kestrel_env.sh" <<ENVEOF
# Source this before running BSM Pixi jobs on Kestrel.
export NLR_ACCOUNT="${NLR_ACCOUNT}"
export KESTREL_PROJECT_ROOT="${KESTREL_PROJECT_ROOT}"
export PIXI_HOME="${PIXI_HOME}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS}"
export PATH="${PIXI_HOME}/bin:\$PATH"
ENVEOF

chmod +x "${OUT_DIR}/pixi_kestrel_env.sh"
kestrel_status "script=install_pixi_project_env.sh step=done_pixi_install log=${LOG} env_file=${OUT_DIR}/pixi_kestrel_env.sh"
echo "Wrote ${LOG} and ${OUT_DIR}/pixi_kestrel_env.sh"
