#!/usr/bin/env bash
set -euo pipefail

echo "RETIRED: use the content-addressed G11 campaign package; this legacy workflow is non-executable." >&2
exit 64

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

STUDY_ID="${STUDY_ID:-publication_full_dataset_distributed_20260519}"  # Original publication run; override: export STUDY_ID=publication_$(date +%Y%m%d)
STUDY_ROOT="${STUDY_ROOT:-/scratch/${USER}/bsm/studies/${STUDY_ID}}"
BASE_CONFIG_REL="${BASE_CONFIG_REL:-configs/hpc/kestrel_publication_full_dataset_distributed_base.yml}"
DATASET_PATH="${DATASET_PATH:-/scratch/${USER}/bsm/bsm-public-rf/artifacts/preprocessed_real_data_30k}"

ARTIFACT_ROOT="${STUDY_ROOT}/artifacts"
SCRIPT_ROOT="${STUDY_ROOT}/hpc_scripts"
LOG_ROOT="${STUDY_ROOT}/logs"
METADATA_ROOT="${STUDY_ROOT}/metadata"
GENERATED_CONFIG_ROOT="${STUDY_ROOT}/generated_configs"
STAGE_JOBS_FILE="${METADATA_ROOT}/stage_jobs.tsv"

export PIXI_HOME="${PIXI_HOME:-/projects/bsm/.pixi}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-/projects/bsm/.cache/pixi}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "${ARTIFACT_ROOT}" "${SCRIPT_ROOT}" "${LOG_ROOT}" "${METADATA_ROOT}" "${GENERATED_CONFIG_ROOT}"

BASE_CONFIG_PATH="${REPO_ROOT}/${BASE_CONFIG_REL}"
if [[ ! -f "${BASE_CONFIG_PATH}" ]]; then
  echo "error: base config not found: ${BASE_CONFIG_PATH}" >&2
  exit 2
fi

for required in X.parquet Y.parquet holdout_assignments.parquet actual_input_feature_catalog.parquet; do
  if [[ ! -f "${DATASET_PATH}/${required}" ]]; then
    echo "error: missing required dataset artifact: ${DATASET_PATH}/${required}" >&2
    exit 2
  fi
done

STAGES=(
  output_conditioning
  empirical_null_screening
  interaction_discovery
  nonlinear_discovery
  sparse_selection
  final_manuscript_artifacts
)

N_SHARDS=(8 800 4000 1600 1200 1600)
N_JOBS=(104 104 104 104 104 104)
CPUS_PER_TASK=(104 104 104 104 104 104)
MEMORY_GB=(230 230 230 230 230 230)
WALLTIME=("01:00:00" "03:00:00" "04:00:00" "03:50:00" "03:50:00" "03:50:00")
MAX_CONCURRENT=(8 1024 4000 1600 1200 1600)
REDUCE_WALLTIME=("00:45:00" "01:30:00" "03:30:00" "02:00:00" "02:30:00" "03:30:00")
REDUCE_MEMORY_GB=(128 128 128 128 128 128)
PARTITIONS=(short short short short short short)
STAGE_QOS="${STAGE_QOS:-}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-}"
QOS_FLAG=""
[[ -n "${STAGE_QOS}" ]] && QOS_FLAG="--qos=${STAGE_QOS}"
RUN_ID_SUFFIX=(s01_output s02_empirical s03_interaction s04_nonlinear s05_sparse s06_final)

# Guard against parallel-array drift: every resource list MUST have the
# same length as STAGES so the for-i loop assigns shards / walltime /
# resources to the right stage. A single edit to one array (e.g.
# adding a stage to STAGES without extending the others) would
# otherwise silently misassign every later stage's resources.
n_stages="${#STAGES[@]}"
for arr_name in N_SHARDS N_JOBS CPUS_PER_TASK MEMORY_GB WALLTIME MAX_CONCURRENT \
                REDUCE_WALLTIME REDUCE_MEMORY_GB PARTITIONS RUN_ID_SUFFIX; do
  declare -n _arr="${arr_name}"
  if [[ "${#_arr[@]}" -ne "${n_stages}" ]]; then
    echo "controller: FATAL: resource array ${arr_name} length ${#_arr[@]} != STAGES length ${n_stages}; " \
         "edit all per-stage arrays together." >&2
    exit 2
  fi
  unset -n _arr
done

echo -e "stage\tarray_job_id\treduce_job_id\tstatus" > "${STAGE_JOBS_FILE}"

wait_for_job() {
  local job_id="$1"
  local label="$2"
  local state=""
  # Bail out if both sacct AND squeue stop returning information for the
  # job after a reasonable grace period — that means the job vanished
  # without recording a terminal state and we'd otherwise loop forever.
  local empty_polls=0
  local max_empty_polls="${CONTROLLER_MAX_EMPTY_POLLS:-40}"  # 40 × 30s = 20 min
  while true; do
    state="$(
      sacct -X -n -j "${job_id}" --format=State 2>/dev/null \
        | head -n 1 \
        | awk '{print $1}'
    )"
    if [[ -z "${state}" ]]; then
      if [[ -n "$(squeue -h -j "${job_id}" 2>/dev/null)" ]]; then
        empty_polls=0
        sleep 30
        continue
      fi
      empty_polls=$((empty_polls + 1))
      if (( empty_polls >= max_empty_polls )); then
        echo "[controller] ${label} disappeared from sacct+squeue for ${empty_polls} polls" >&2
        echo "[controller]   (job=${job_id}); aborting wait. Inspect cluster manually." >&2
        return 2
      fi
      sleep 30
      continue
    fi
    empty_polls=0
    case "${state}" in
      COMPLETED*)
        echo "[controller] ${label} completed (job=${job_id})"
        return 0
        ;;
      FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)
        echo "[controller] ${label} failed (job=${job_id}, state=${state})" >&2
        return 1
        ;;
      *)
        sleep 60
        ;;
    esac
  done
}

generate_stage_config() {
  local run_id="$1"
  local n_jobs="$2"
  local cpus_per_task="$3"
  local memory_gb="$4"
  local walltime="$5"
  local max_concurrent="$6"
  local partition="$7"
  local stage_log_dir="$8"
  local dataset_path="$9"
  local output_path="${10}"
  local config_out="${11}"

  pixi run python - \
    "${BASE_CONFIG_PATH}" \
    "${config_out}" \
    "${run_id}" \
    "${n_jobs}" \
      "${cpus_per_task}" \
      "${memory_gb}" \
      "${walltime}" \
      "${max_concurrent}" \
      "${partition}" \
      "${stage_log_dir}" \
      "${dataset_path}" \
      "${output_path}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

run_id = sys.argv[3]
n_jobs = int(sys.argv[4])
cpus_per_task = int(sys.argv[5])
memory_gb = int(sys.argv[6])
walltime = sys.argv[7]
max_concurrent = int(sys.argv[8])
partition = sys.argv[9]
stage_log_dir = sys.argv[10]
dataset_path = sys.argv[11]
output_path = sys.argv[12]

raw = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
if not isinstance(raw, dict):
    raise SystemExit(f"invalid base config: {base_path}")

raw.setdefault("runtime", {})
raw.setdefault("dataset", {})
raw.setdefault("output", {})
raw.setdefault("distributed", {})
raw["runtime"]["n_jobs"] = n_jobs
raw["dataset"]["path"] = dataset_path
raw["output"]["artifact_dir"] = output_path
raw["distributed"]["enabled"] = True
raw["distributed"]["backend"] = "slurm_array"
raw["distributed"]["run_id"] = run_id
slurm = raw["distributed"].setdefault("slurm", {})
slurm["cpus_per_task"] = cpus_per_task
slurm["memory_gb"] = memory_gb
slurm["walltime"] = walltime
slurm["max_concurrent_array_tasks"] = max_concurrent
slurm["partition"] = partition
slurm["log_dir"] = stage_log_dir

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
PY
}

submit_stage() {
  local stage="$1"
  local idx="$2"
  local previous_reduce_job_id="$3"

  local n_shards="${N_SHARDS[idx]}"
  local n_jobs="${N_JOBS[idx]}"
  local cpus_per_task="${CPUS_PER_TASK[idx]}"
  local memory_gb="${MEMORY_GB[idx]}"
  local walltime="${WALLTIME[idx]}"
  local max_concurrent="${MAX_CONCURRENT[idx]}"
  local reduce_walltime="${REDUCE_WALLTIME[idx]}"
  local reduce_memory_gb="${REDUCE_MEMORY_GB[idx]}"
  local partition="${PARTITIONS[idx]}"
  local run_suffix="${RUN_ID_SUFFIX[idx]}"

  local run_id="bsm_${STUDY_ID}_${run_suffix}"
  local stage_log_dir="/scratch/${USER}/bsm/${run_id}/logs"
  local stage_config="${GENERATED_CONFIG_ROOT}/${stage}.yml"
  local stage_script_dir="${SCRIPT_ROOT}/${stage}"
  local stage_shard_dir="${ARTIFACT_ROOT}/hpc_shards_${stage}"
  local active_shard_symlink="${ARTIFACT_ROOT}/hpc_shards"

  rm -rf "${stage_script_dir}"
  mkdir -p "${stage_script_dir}" "${stage_shard_dir}" "${stage_log_dir}"
  ln -sfn "${stage_shard_dir}" "${active_shard_symlink}"

  generate_stage_config \
    "${run_id}" \
    "${n_jobs}" \
    "${cpus_per_task}" \
    "${memory_gb}" \
    "${walltime}" \
    "${max_concurrent}" \
    "${partition}" \
    "${stage_log_dir}" \
    "${DATASET_PATH}" \
    "${ARTIFACT_ROOT}" \
    "${stage_config}"

  local shard_args=()
  if [[ "${n_shards}" -gt 0 ]]; then
    shard_args=(--n-shards "${n_shards}")
  fi

  echo "[controller] generating scripts: stage=${stage} shards=${n_shards}"
  (
    cd "${REPO_ROOT}" && \
      pixi run rfm-hpc-submit \
        --config "${stage_config}" \
        --stage "${stage}" \
        "${shard_args[@]}" \
        --output-dir "${stage_script_dir}" \
        --reduce-walltime "${reduce_walltime}" \
        --reduce-memory-gb "${reduce_memory_gb}"
  )

  local array_script="${stage_script_dir}/submit_${stage}_array.sh"
  local reduce_script="${stage_script_dir}/submit_${stage}_reduce.sh"
  local array_job_id=""
  local reduce_job_id=""

  if [[ -f "${array_script}" ]]; then
    if [[ -n "${previous_reduce_job_id}" ]]; then
      array_job_id="$(
        sbatch --parsable ${QOS_FLAG} --dependency=afterok:"${previous_reduce_job_id}" "${array_script}"
      )"
    else
      array_job_id="$(sbatch --parsable ${QOS_FLAG} "${array_script}")"
    fi
    reduce_job_id="$(sbatch --parsable ${QOS_FLAG} --dependency=afterany:"${array_job_id}" "${reduce_script}")"
  else
    if [[ -n "${previous_reduce_job_id}" ]]; then
      reduce_job_id="$(
        sbatch --parsable ${QOS_FLAG} --dependency=afterok:"${previous_reduce_job_id}" "${reduce_script}"
      )"
    else
      reduce_job_id="$(sbatch --parsable ${QOS_FLAG} "${reduce_script}")"
    fi
  fi

  echo -e "${stage}\t${array_job_id}\t${reduce_job_id}\tsubmitted" >> "${STAGE_JOBS_FILE}"
  echo "[controller] submitted stage=${stage} array_job=${array_job_id:-none} reduce_job=${reduce_job_id}"

  if ! wait_for_job "${reduce_job_id}" "${stage} reduce"; then
    echo -e "${stage}\t${array_job_id}\t${reduce_job_id}\tfailed" >> "${STAGE_JOBS_FILE}"
    echo "${stage}" > "${METADATA_ROOT}/RUN_FAILED_STAGE.txt"
    return 1
  fi
  echo -e "${stage}\t${array_job_id}\t${reduce_job_id}\tcompleted" >> "${STAGE_JOBS_FILE}"
  STAGE_REDUCE_JOB_ID="${reduce_job_id}"
}

echo "[controller] study_id=${STUDY_ID}"
echo "[controller] study_root=${STUDY_ROOT}"
echo "[controller] repo_root=${REPO_ROOT}"
echo "[controller] dataset_path=${DATASET_PATH}"

cd "${REPO_ROOT}"
pixi install --locked

previous_reduce_job_id=""
STAGE_REDUCE_JOB_ID=""
START_AT_STAGE="${START_AT_STAGE:-}"
_started=0
[[ -z "${START_AT_STAGE}" ]] && _started=1
for i in "${!STAGES[@]}"; do
  stage="${STAGES[i]}"
  if [[ "${_started}" -eq 0 ]]; then
    if [[ "${stage}" == "${START_AT_STAGE}" ]]; then
      _started=1
    else
      echo "[controller] START_AT_STAGE=${START_AT_STAGE}: skipping completed stage=${stage}"
      continue
    fi
  fi
  if ! submit_stage "${stage}" "${i}" "${previous_reduce_job_id}"; then
    echo "[controller] pipeline failed at stage=${stage}" >&2
    exit 1
  fi
  previous_reduce_job_id="${STAGE_REDUCE_JOB_ID}"
  if [[ -n "${STOP_AFTER_STAGE}" && "${stage}" == "${STOP_AFTER_STAGE}" ]]; then
    echo "[controller] STOP_AFTER_STAGE=${STOP_AFTER_STAGE} reached; pausing before remaining stages"
    touch "${METADATA_ROOT}/RUN_PAUSED_AFTER_${stage}"
    exit 0
  fi
done

touch "${METADATA_ROOT}/RUN_COMPLETE"
echo "[controller] full distributed pipeline complete"
