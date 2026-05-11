#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=../hpc_kestrel_config.sh
source "${REPO_ROOT}/scripts/hpc_kestrel_config.sh"

cd "${REPO_ROOT}"

RUN_ROOT="${1:-${REPO_ROOT}/artifacts/hpc_final_cost_ladder}"
RUN_ROOT="$(kestrel_realpath "${RUN_ROOT}")"
mkdir -p "${RUN_ROOT}"

# Prefer scratch spill for Kestrel CPU nodes.
export TMPDIR="${TMPDIR:-/scratch/${USER}/bsm_final_cost/${SLURM_JOB_ID:-manual}}"
mkdir -p "${TMPDIR}"

# Keep BLAS/OpenMP pinned to one thread per process; let n_jobs control parallelism.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BSM_PROGRESS_BATCH_SIZE=1

run_pipeline() {
  local cfg="$1"
  shift
  echo ">>> $(date -u +%Y-%m-%dT%H:%M:%SZ) RUN config=${cfg} args=$*"
  pixi run python tools/run_manuscript_pipeline.py "${cfg}" "$@"
}

echo ">>> precompute base artifacts through nonlinear_discovery"
run_pipeline configs/kestrel_final_cost_base_precompute.yml --stop-stage nonlinear_discovery

copy_base() {
  local out_dir="$1"
  mkdir -p "${out_dir}"
  for stage in output_conditioning empirical_null_screen interaction_discovery nonlinear_discovery; do
    if [[ ! -d "${out_dir}/${stage}" ]]; then
      cp -R "${RUN_ROOT}/base/${stage}" "${out_dir}/${stage}"
    fi
  done
}

run_ladder() {
  local label="$1"
  local cfg="$2"
  local out_dir="${RUN_ROOT}/${label}"
  copy_base "${out_dir}"
  echo ">>> run sparse->final label=${label}"
  run_pipeline "${cfg}" --start-stage sparse_selection --output-dir "${out_dir}"
}

run_ladder 01 configs/kestrel_final_cost_sparse_final_01.yml
run_ladder 02 configs/kestrel_final_cost_sparse_final_02.yml
run_ladder 03 configs/kestrel_final_cost_sparse_final_03.yml
run_ladder 04 configs/kestrel_final_cost_sparse_final_04.yml
run_ladder 05_near_uncapped configs/kestrel_final_cost_sparse_final_05_near_uncapped.yml

summary="${RUN_ROOT}/summary.csv"
printf 'run_id,complete,failed,elapsed_seconds,sparse_seconds,final_seconds,n_prefilter,n_final,max_candidate_terms,bootstrap_count\n' > "${summary}"
for pair in \
  "01 configs/kestrel_final_cost_sparse_final_01.yml" \
  "02 configs/kestrel_final_cost_sparse_final_02.yml" \
  "03 configs/kestrel_final_cost_sparse_final_03.yml" \
  "04 configs/kestrel_final_cost_sparse_final_04.yml" \
  "05_near_uncapped configs/kestrel_final_cost_sparse_final_05_near_uncapped.yml"; do
  run_id="${pair%% *}"
  cfg="${pair#* }"
  out_dir="${RUN_ROOT}/${run_id}"
  complete=no
  failed=no
  elapsed=""
  [[ -f "${out_dir}/run_complete.json" ]] && complete=yes
  [[ -f "${out_dir}/run_failed.json" ]] && failed=yes
  if [[ -f "${out_dir}/run_complete.json" ]]; then
    elapsed="$(rg '"elapsed_seconds"' "${out_dir}/run_complete.json" | sed -E 's/.*: ([0-9.]+).*/\1/')"
  fi
  rt="${out_dir}/runtime_diagnostics/stage_runtime_summary.csv"
  sparse_seconds=""
  final_seconds=""
  if [[ -f "${rt}" ]]; then
    sparse_seconds="$(awk -F, '$1=="sparse_selection_and_stability"{print $2}' "${rt}" | head -n1)"
    final_seconds="$(awk -F, '$1=="final_manuscript_tables_and_figures"{print $2}' "${rt}" | head -n1)"
  fi
  final_summary="${out_dir}/final_manuscript_artifacts/final_artifact_summary.csv"
  n_prefilter=""
  n_final=""
  if [[ -f "${final_summary}" ]]; then
    n_prefilter="$(awk -F, 'NR==2{print $2}' "${final_summary}")"
    n_final="$(awk -F, 'NR==2{print $3}' "${final_summary}")"
  fi
  max_candidate_terms="$(awk '/max_candidate_terms:/{print $2}' "${cfg}" | head -n1)"
  bootstrap_count="$(awk '/bootstrap_count:/{print $2}' "${cfg}" | head -n1)"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${run_id}" "${complete}" "${failed}" "${elapsed}" "${sparse_seconds}" "${final_seconds}" \
    "${n_prefilter}" "${n_final}" "${max_candidate_terms}" "${bootstrap_count}" >> "${summary}"
done

echo ">>> summary=${summary}"
cat "${summary}"
