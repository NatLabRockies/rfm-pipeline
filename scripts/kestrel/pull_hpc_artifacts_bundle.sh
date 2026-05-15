#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=common_paths.sh
source "${REPO_ROOT}/scripts/kestrel/common_paths.sh"

HPC_HOST="${HPC_HOST:-kl1.hpc.nrel.gov}"
HPC_REPO_ROOT="${HPC_REPO_ROOT:-/projects/bsm/bsm-public-rf}"
HPC_ARTIFACTS_ROOT="${HPC_ARTIFACTS_ROOT:-__AUTO__}"
LOCAL_OUT_DIR="${LOCAL_OUT_DIR:-${REPO_ROOT}/artifacts/kestrel_collected_bundles}"
REMOTE_SNAPSHOT_ROOT="${REMOTE_SNAPSHOT_ROOT:-}"
PULLBACK_MODE="${PULLBACK_MODE:-reporting_bundle}"
CPU_TIER_SPECS="${CPU_TIER_SPECS:-2=configs/hpc/kestrel_cpu_scale_2.yml,10=configs/hpc/kestrel_cpu_scale_10.yml,1000=configs/hpc/kestrel_cpu_scale_1000.yml}"
INCLUDE_GPU="${INCLUDE_GPU:-1}"
GPU_CONFIG_PATH="${GPU_CONFIG_PATH:-configs/hpc/kestrel_gpu_h100.yml}"
KEEP_REMOTE=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/kestrel/pull_hpc_artifacts_bundle.sh [options]

Pull HPC run artifacts by:
  1) building a manifest and snapshot bundle on Kestrel
  2) downloading the zip locally
  3) writing a local analysis CSV/TXT from the embedded manifest

Options:
  --hpc-host HOST            SSH host/alias for Kestrel (default: $HPC_HOST)
  --hpc-repo-root DIR        Repo root on HPC (default: $HPC_REPO_ROOT)
  --hpc-artifacts-root DIR   Artifact root on HPC (default: auto; prefers /scratch/$USER/bsm/bsm-public-rf/artifacts)
  --local-out-dir DIR        Local destination for bundles (default: ./artifacts/kestrel_collected_bundles)
  --remote-snapshot-root DIR Remote snapshot root (default: auto)
  --pullback-mode MODE       manifest_only | reporting_bundle | full | study_package (default: reporting_bundle)
  --cpu-tier-specs SPECS     Comma list: <nodes>=<config-path> (default: 2/10/1000 configs)
  --include-gpu 0|1          Include GPU target in manifest/bundle (default: 1)
  --gpu-config PATH          GPU config path relative to remote repo root
  --keep-remote              Keep remote snapshot dir + zip after download
  -h, --help                 Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hpc-host)
      HPC_HOST="${2:-}"
      shift 2
      ;;
    --hpc-repo-root)
      HPC_REPO_ROOT="${2:-}"
      shift 2
      ;;
    --hpc-artifacts-root)
      HPC_ARTIFACTS_ROOT="${2:-}"
      shift 2
      ;;
    --local-out-dir)
      LOCAL_OUT_DIR="${2:-}"
      shift 2
      ;;
    --remote-snapshot-root)
      REMOTE_SNAPSHOT_ROOT="${2:-}"
      shift 2
      ;;
    --pullback-mode)
      PULLBACK_MODE="${2:-}"
      shift 2
      ;;
    --cpu-tier-specs)
      CPU_TIER_SPECS="${2:-}"
      shift 2
      ;;
    --include-gpu)
      INCLUDE_GPU="${2:-}"
      shift 2
      ;;
    --gpu-config)
      GPU_CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --keep-remote)
      KEEP_REMOTE=1
      shift
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

case "${PULLBACK_MODE}" in
  manifest_only|reporting_bundle|full|study_package)
    ;;
  *)
    echo "error: --pullback-mode must be one of: manifest_only, reporting_bundle, full, study_package" >&2
    exit 2
    ;;
esac

if [[ "${INCLUDE_GPU}" != "0" && "${INCLUDE_GPU}" != "1" ]]; then
  echo "error: --include-gpu must be 0 or 1" >&2
  exit 2
fi

if [[ -z "${HPC_HOST}" || -z "${HPC_REPO_ROOT}" || -z "${LOCAL_OUT_DIR}" ]]; then
  echo "error: host/repo-root/local-out-dir cannot be empty" >&2
  exit 2
fi

mkdir -p "${LOCAL_OUT_DIR}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_SNAPSHOT_ARG="${REMOTE_SNAPSHOT_ROOT:-__AUTO__}"
REMOTE_OUTPUT_FILE="$(mktemp "${TMPDIR:-/tmp}/bsm_hpc_pull.XXXXXX")"
trap 'rm -f "${REMOTE_OUTPUT_FILE}"' EXIT

echo "==> Building remote bundle on ${HPC_HOST} (mode=${PULLBACK_MODE})"
ssh -T "${HPC_HOST}" "bash -s" -- \
  "${HPC_REPO_ROOT}" \
  "${TS}" \
  "${REMOTE_SNAPSHOT_ARG}" \
  "${HPC_ARTIFACTS_ROOT}" \
  "${PULLBACK_MODE}" \
  "${CPU_TIER_SPECS}" \
  "${INCLUDE_GPU}" \
  "${GPU_CONFIG_PATH}" >"${REMOTE_OUTPUT_FILE}" <<'REMOTE_EOF'
set -euo pipefail

HPC_REPO_ROOT="$1"
TS="$2"
REMOTE_SNAPSHOT_ROOT="${3:-__AUTO__}"
HPC_ARTIFACTS_ROOT="${4:-__AUTO__}"
PULLBACK_MODE="${5:-reporting_bundle}"
CPU_TIER_SPECS="${6:-}"
INCLUDE_GPU="${7:-1}"
GPU_CONFIG_PATH="${8:-configs/hpc/kestrel_gpu_h100.yml}"

if [[ "${REMOTE_SNAPSHOT_ROOT}" == "__AUTO__" ]]; then
  REMOTE_SNAPSHOT_ROOT=""
fi
case "${PULLBACK_MODE}" in
  manifest_only|reporting_bundle|full|study_package)
    ;;
  *)
    echo "error: invalid pullback mode: ${PULLBACK_MODE}" >&2
    exit 2
    ;;
esac

if [[ "${INCLUDE_GPU}" != "0" && "${INCLUDE_GPU}" != "1" ]]; then
  echo "error: INCLUDE_GPU must be 0 or 1" >&2
  exit 2
fi

if [[ ! -d "${HPC_REPO_ROOT}" ]]; then
  echo "error: HPC repo root not found: ${HPC_REPO_ROOT}" >&2
  exit 2
fi
cd "${HPC_REPO_ROOT}"

REMOTE_USER="${USER:-}"
if [[ -z "${REMOTE_USER}" ]]; then
  REMOTE_USER="$(id -un 2>/dev/null || true)"
fi
if [[ -z "${REMOTE_USER}" ]]; then
  echo "error: unable to resolve remote username for scratch/log path defaults" >&2
  exit 2
fi
if [[ "${HPC_ARTIFACTS_ROOT}" == "__AUTO__" ]]; then
  SCRATCH_ARTIFACTS_ROOT="/scratch/${REMOTE_USER}/bsm/bsm-public-rf/artifacts"
  if [[ -d "${SCRATCH_ARTIFACTS_ROOT}" || -L "${SCRATCH_ARTIFACTS_ROOT}" ]]; then
    HPC_ARTIFACTS_ROOT="${SCRATCH_ARTIFACTS_ROOT}"
  else
    HPC_ARTIFACTS_ROOT="${HPC_REPO_ROOT}/artifacts"
  fi
fi
if [[ "${HPC_ARTIFACTS_ROOT}" == "/home/${REMOTE_USER}/"* ]]; then
  echo "error: refusing home-directory artifacts root: ${HPC_ARTIFACTS_ROOT}" >&2
  exit 2
fi

if [[ -n "${REMOTE_SNAPSHOT_ROOT}" ]]; then
  SNAPSHOT_ROOT="${REMOTE_SNAPSHOT_ROOT}"
elif [[ "${HPC_REPO_ROOT}" == "/home/${REMOTE_USER}/"* ]]; then
  SNAPSHOT_ROOT="/scratch/${REMOTE_USER}/bsm/kestrel_hpc_snapshots"
else
  SNAPSHOT_ROOT="${HPC_REPO_ROOT}/artifacts/kestrel_hpc_snapshots"
fi
if [[ "${SNAPSHOT_ROOT}" == "/home/${REMOTE_USER}/"* ]]; then
  echo "error: refusing home-directory snapshot root: ${SNAPSHOT_ROOT}" >&2
  exit 2
fi

SNAPSHOT_DIR="${SNAPSHOT_ROOT}/${TS}"
BUNDLE_DIR="${SNAPSHOT_DIR}/bundle"
RUNS_DIR="${BUNDLE_DIR}/runs"
LOGS_DIR="${BUNDLE_DIR}/logs"
MANIFEST_DIR="${BUNDLE_DIR}/manifest"
ZIP_PATH="${SNAPSHOT_ROOT}/kestrel_hpc_snapshot_${TS}.zip"
LOGS_ROOT="/scratch/${REMOTE_USER}/bsm"
SUITE_ROOT="${HPC_ARTIFACTS_ROOT}/kestrel_cpu_scaling_suite"
MANIFEST_JSON="${MANIFEST_DIR}/hpc_run_manifest.json"
MANIFEST_CSV="${MANIFEST_DIR}/run_summary.csv"
TARGET_SPECS_JSON="${MANIFEST_DIR}/target_specs.json"
STUDY_METADATA_JSON="${MANIFEST_DIR}/study_metadata_manifest.json"
STUDY_FILE_INVENTORY_CSV="${MANIFEST_DIR}/study_file_inventory.csv"
REPRO_COMMANDS_JSON="${MANIFEST_DIR}/commands.json"
REPRO_RECIPE_MD="${MANIFEST_DIR}/reproduction_recipe.md"

mkdir -p "${RUNS_DIR}" "${LOGS_DIR}" "${MANIFEST_DIR}"

if command -v pixi >/dev/null 2>&1; then
  PYTHON_RUNNER=(pixi run python)
else
  PYTHON_RUNNER=(python3)
fi

"${PYTHON_RUNNER[@]}" - \
  "${HPC_REPO_ROOT}" \
  "${LOGS_ROOT}" \
  "${SUITE_ROOT}" \
  "${CPU_TIER_SPECS}" \
  "${INCLUDE_GPU}" \
  "${GPU_CONFIG_PATH}" \
  "${REMOTE_USER}" \
  "${TARGET_SPECS_JSON}" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

import yaml


def _resolve_path(repo_root: Path, raw: str, default_value: str) -> str:
    value = str(raw or default_value)
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((repo_root / path).resolve())


def _expand_log_dir(template: str, *, run_id: str, remote_user: str) -> str:
    value = str(template)
    value = value.replace("${RUN_ID}", run_id).replace("${USER}", remote_user)
    return os.path.expandvars(value)


repo_root = Path(sys.argv[1]).resolve()
logs_root = Path(sys.argv[2]).resolve()
suite_root = Path(sys.argv[3]).resolve()
cpu_tier_specs = sys.argv[4]
include_gpu = sys.argv[5] == "1"
gpu_config_path = sys.argv[6]
remote_user = sys.argv[7]
out_path = Path(sys.argv[8]).resolve()

targets: list[dict[str, object]] = []
for token in [part.strip() for part in cpu_tier_specs.split(",") if part.strip()]:
    if "=" not in token:
        raise SystemExit(f"Invalid cpu tier token (expected <nodes>=<config>): {token}")
    nodes_raw, cfg_raw = token.split("=", 1)
    nodes = int(nodes_raw.strip())
    cfg_path = Path(cfg_raw.strip())
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    output = cfg.get("output", {}) or {}
    distributed = cfg.get("distributed", {}) or {}
    slurm = distributed.get("slurm", {}) or {}
    run_id = str(distributed.get("run_id", f"bsm_kestrel_cpu_scale_{nodes}"))
    run_dir = _resolve_path(
        repo_root,
        str(output.get("artifact_dir", f"./artifacts/kestrel_cpu_scale_{nodes}_run")),
        f"./artifacts/kestrel_cpu_scale_{nodes}_run",
    )
    log_dir = _expand_log_dir(
        str(slurm.get("log_dir", str(logs_root / run_id / "logs"))),
        run_id=run_id,
        remote_user=remote_user,
    )
    suite_manifest = str((suite_root / f"cpu_nodes_{nodes}" / "hpc_scripts" / "manifest.jsonl").resolve())
    targets.append(
        {
            "target": f"cpu_{nodes}",
            "kind": "cpu",
            "nodes": nodes,
            "config_path": str(cfg_path),
            "run_id": run_id,
            "run_dir": run_dir,
            "log_dir": log_dir,
            "suite_manifest_path": suite_manifest,
            "gpu_mode": False,
        }
    )

if include_gpu:
    gpu_cfg_path = Path(gpu_config_path)
    if not gpu_cfg_path.is_absolute():
        gpu_cfg_path = (repo_root / gpu_cfg_path).resolve()
    gpu_cfg = yaml.safe_load(gpu_cfg_path.read_text(encoding="utf-8")) or {}
    output = gpu_cfg.get("output", {}) or {}
    distributed = gpu_cfg.get("distributed", {}) or {}
    slurm = distributed.get("slurm", {}) or {}
    run_id = str(distributed.get("run_id", "bsm_kestrel_gpu_h100"))
    run_dir = _resolve_path(
        repo_root,
        str(output.get("artifact_dir", "./artifacts/kestrel_gpu_h100_run")),
        "./artifacts/kestrel_gpu_h100_run",
    )
    log_dir = _expand_log_dir(
        str(slurm.get("log_dir", str(logs_root / run_id / "logs"))),
        run_id=run_id,
        remote_user=remote_user,
    )
    targets.append(
        {
            "target": "gpu_h100",
            "kind": "gpu",
            "nodes": 0,
            "config_path": str(gpu_cfg_path),
            "run_id": run_id,
            "run_dir": run_dir,
            "log_dir": log_dir,
            "suite_manifest_path": "",
            "gpu_mode": True,
        }
    )

payload = {
    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hpc_repo_root": str(repo_root),
    "cpu_tier_specs": cpu_tier_specs,
    "include_gpu": include_gpu,
    "gpu_config_path": str((repo_root / gpu_config_path).resolve()),
    "targets": targets,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

"${PYTHON_RUNNER[@]}" "${HPC_REPO_ROOT}/tools/hpc_bundle_manifest.py" create-run-manifest \
  --hpc-repo-root "${HPC_REPO_ROOT}" \
  --artifacts-root "${HPC_ARTIFACTS_ROOT}" \
  --logs-root "${LOGS_ROOT}" \
  --suite-root "${SUITE_ROOT}" \
  --pullback-mode "${PULLBACK_MODE}" \
  --target-specs-json "${TARGET_SPECS_JSON}" \
  --output-json "${MANIFEST_JSON}" \
  --output-csv "${MANIFEST_CSV}"

copy_tree() {
  local src="$1"
  local dst="$2"
  if [[ ! -d "${src}" ]]; then
    return 0
  fi
  mkdir -p "${dst}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${src}/" "${dst}/"
  else
    cp -a "${src}/." "${dst}/"
  fi
}

copy_file_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "${src}" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${dst}")"
  cp -f "${src}" "${dst}"
}

copy_latest_match() {
  local pattern="$1"
  local dst_dir="$2"
  local latest
  latest="$(ls -1t ${pattern} 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest}" && -f "${latest}" ]]; then
    mkdir -p "${dst_dir}"
    cp -f "${latest}" "${dst_dir}/"
  fi
}

if [[ "${PULLBACK_MODE}" == "full" || "${PULLBACK_MODE}" == "reporting_bundle" || "${PULLBACK_MODE}" == "study_package" ]]; then
  copy_tree "${SUITE_ROOT}" "${RUNS_DIR}/cpu_scaling_suite"
fi

if [[ "${PULLBACK_MODE}" == "study_package" ]]; then
  copy_file_if_exists "${HPC_REPO_ROOT}/pixi.lock" "${MANIFEST_DIR}/environment/pixi.lock"
  copy_file_if_exists "${HPC_REPO_ROOT}/pixi.toml" "${MANIFEST_DIR}/environment/pixi.toml"
  copy_file_if_exists "${HPC_REPO_ROOT}/pyproject.toml" "${MANIFEST_DIR}/environment/pyproject.toml"
fi

while IFS=$'\t' read -r target kind run_dir log_dir suite_manifest config_path gpu_mode; do
  [[ -z "${target}" ]] && continue
  out_run="${RUNS_DIR}/${target}"
  out_logs="${LOGS_DIR}/${target}"
  if [[ "${PULLBACK_MODE}" == "reporting_bundle" ]]; then
    copy_tree "${run_dir}/hpc_scripts" "${out_run}/hpc_scripts"
    copy_tree "${run_dir}/hpc_shards/_merged" "${out_run}/hpc_shards/_merged"
    if [[ -n "${suite_manifest}" ]]; then
      copy_tree "$(dirname "${suite_manifest}")" "${out_run}/hpc_scripts"
      copy_file_if_exists "${suite_manifest}" "${out_run}/hpc_scripts/suite_manifest.jsonl"
    fi
    if [[ "${gpu_mode}" == "1" ]]; then
      copy_latest_match "${log_dir}/bsm_gpu_interaction_discovery_*.out" "${out_logs}"
    fi
    copy_latest_match "${log_dir}/bsm_interaction_discovery_*.out" "${out_logs}"
    copy_latest_match "${log_dir}/bsm_reduce_interaction_discovery_*.out" "${out_logs}"
  elif [[ "${PULLBACK_MODE}" == "full" ]]; then
    copy_tree "${run_dir}/hpc_scripts" "${out_run}/hpc_scripts"
    copy_tree "${run_dir}/hpc_shards" "${out_run}/hpc_shards"
    copy_tree "${log_dir}" "${out_logs}"
    if [[ -n "${suite_manifest}" ]]; then
      copy_tree "$(dirname "${suite_manifest}")" "${out_run}/hpc_scripts"
      copy_file_if_exists "${suite_manifest}" "${out_run}/hpc_scripts/suite_manifest.jsonl"
    fi
  elif [[ "${PULLBACK_MODE}" == "study_package" ]]; then
    copy_tree "${run_dir}" "${out_run}/run_artifacts"
    copy_tree "${log_dir}" "${out_logs}"
    if [[ -n "${suite_manifest}" ]]; then
      copy_tree "$(dirname "${suite_manifest}")" "${out_run}/hpc_scripts"
      copy_file_if_exists "${suite_manifest}" "${out_run}/hpc_scripts/suite_manifest.jsonl"
    fi
    if [[ -n "${config_path}" ]]; then
      copy_file_if_exists "${config_path}" "${MANIFEST_DIR}/configs/${target}.yml"
    fi
  fi
done < <("${PYTHON_RUNNER[@]}" - "${TARGET_SPECS_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for target in payload.get("targets", []):
    row = [
        str(target.get("target", "")),
        str(target.get("kind", "")),
        str(target.get("run_dir", "")),
        str(target.get("log_dir", "")),
        str(target.get("suite_manifest_path", "")),
        str(target.get("config_path", "")),
        "1" if bool(target.get("gpu_mode", False)) else "0",
    ]
    print("\t".join(row))
PY
)

"${PYTHON_RUNNER[@]}" "${HPC_REPO_ROOT}/tools/hpc_bundle_manifest.py" write-study-metadata \
  --bundle-root "${BUNDLE_DIR}" \
  --hpc-repo-root "${HPC_REPO_ROOT}" \
  --target-specs-json "${TARGET_SPECS_JSON}" \
  --pullback-mode "${PULLBACK_MODE}" \
  --output-json "${STUDY_METADATA_JSON}" \
  --output-csv "${STUDY_FILE_INVENTORY_CSV}" \
  --commands-json "${REPRO_COMMANDS_JSON}" \
  --reproduction-recipe-md "${REPRO_RECIPE_MD}"

"${PYTHON_RUNNER[@]}" - "${BUNDLE_DIR}" "${ZIP_PATH}" <<'PY'
import os
import sys
import zipfile
from pathlib import Path

bundle_dir = Path(sys.argv[1]).resolve()
zip_path = Path(sys.argv[2]).resolve()
zip_path.parent.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(str(bundle_dir)):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            arcname = os.path.relpath(fpath, str(bundle_dir))
            zf.write(fpath, arcname=arcname)
PY

echo "REMOTE_ZIP_PATH=${ZIP_PATH}"
echo "REMOTE_SNAPSHOT_DIR=${SNAPSHOT_DIR}"
echo "REMOTE_MANIFEST_JSON=${MANIFEST_JSON}"
echo "REMOTE_MANIFEST_CSV=${MANIFEST_CSV}"
REMOTE_EOF

REMOTE_OUTPUT="$(cat "${REMOTE_OUTPUT_FILE}")"

REMOTE_ZIP_PATH="$(printf '%s\n' "${REMOTE_OUTPUT}" | awk -F= '/^REMOTE_ZIP_PATH=/{print $2}' | tail -n 1)"
REMOTE_SNAPSHOT_DIR="$(printf '%s\n' "${REMOTE_OUTPUT}" | awk -F= '/^REMOTE_SNAPSHOT_DIR=/{print $2}' | tail -n 1)"

if [[ -z "${REMOTE_ZIP_PATH}" ]]; then
  echo "error: failed to parse REMOTE_ZIP_PATH from ssh output" >&2
  printf '%s\n' "${REMOTE_OUTPUT}" >&2
  exit 1
fi

echo "==> Downloading ${REMOTE_ZIP_PATH}"
scp "${HPC_HOST}:${REMOTE_ZIP_PATH}" "${LOCAL_OUT_DIR}/"

LOCAL_ZIP_PATH="${LOCAL_OUT_DIR}/$(basename "${REMOTE_ZIP_PATH}")"
LOCAL_ANALYSIS_CSV="${LOCAL_ZIP_PATH%.zip}_analysis.csv"
LOCAL_ANALYSIS_TXT="${LOCAL_ZIP_PATH%.zip}_analysis.txt"

if command -v pixi >/dev/null 2>&1; then
  PY_RUNNER=(pixi run python)
else
  PY_RUNNER=(python3)
fi

"${PY_RUNNER[@]}" "${REPO_ROOT}/tools/hpc_bundle_manifest.py" analyze-zip \
  --zip "${LOCAL_ZIP_PATH}" \
  --out-csv "${LOCAL_ANALYSIS_CSV}" \
  --out-txt "${LOCAL_ANALYSIS_TXT}"

if [[ "${KEEP_REMOTE}" -eq 0 ]]; then
  echo "==> Cleaning remote temporary bundle"
  ssh -T "${HPC_HOST}" "bash -s" -- "${REMOTE_ZIP_PATH}" "${REMOTE_SNAPSHOT_DIR}" <<'REMOTE_CLEAN_EOF'
set -euo pipefail
zip_path="$1"
snapshot_dir="$2"
[[ -n "${zip_path}" ]] && rm -f "${zip_path}" || true
[[ -n "${snapshot_dir}" ]] && rm -rf "${snapshot_dir}" || true
REMOTE_CLEAN_EOF
fi

echo "==> Done"
echo "Local bundle: ${LOCAL_ZIP_PATH}"
echo "Local analysis CSV: ${LOCAL_ANALYSIS_CSV}"
echo "Local analysis TXT: ${LOCAL_ANALYSIS_TXT}"
