#!/usr/bin/env bash
set -euo pipefail

# Temporary script: scp GPU prerequisite artifacts to Kestrel
# Not committed; delete after running

HOST="${HPC_HOST:-kl1.hpc.nrel.gov}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/projects/bsm/bsm-public-rf}"
REMOTE_ARTIFACTS_ROOT="${REMOTE_ARTIFACTS_ROOT:-/scratch/${USER}/bsm/bsm-public-rf/artifacts}"
TARGET_ROOT="${TARGET_ROOT:-${REMOTE_ARTIFACTS_ROOT}/kestrel_gpu_h100_run}"

echo "Creating target directories on Kestrel..."
ssh "$HOST" "mkdir -p '${TARGET_ROOT}/output_conditioning' '${TARGET_ROOT}/empirical_null_screen'"

echo "Copying output_conditioning artifacts..."
scp artifacts/final_cost_ladder/04/output_conditioning/pca_scores.csv "${HOST}:${TARGET_ROOT}/output_conditioning/"
scp artifacts/final_cost_ladder/04/output_conditioning/pca_loadings.csv "${HOST}:${TARGET_ROOT}/output_conditioning/"
scp artifacts/final_cost_ladder/04/output_conditioning/pca_explained_variance.csv "${HOST}:${TARGET_ROOT}/output_conditioning/"

echo "Copying empirical_null_screen artifacts..."
scp artifacts/final_cost_ladder/04/empirical_null_screen/retained_terms.csv "${HOST}:${TARGET_ROOT}/empirical_null_screen/"
scp artifacts/final_cost_ladder/04/empirical_null_screen/component_coefficients.csv "${HOST}:${TARGET_ROOT}/empirical_null_screen/"

echo "Copying interaction_discovery inputs..."
scp artifacts/test_dataset_3k/X.parquet "${HOST}:${TARGET_ROOT}/"
scp artifacts/test_dataset_3k/holdout_assignments.parquet "${HOST}:${TARGET_ROOT}/"
scp artifacts/test_dataset_3k/actual_input_feature_catalog.parquet "${HOST}:${TARGET_ROOT}/"

echo "Complete. GPU artifacts ready on Kestrel at $TARGET_ROOT"
