#!/usr/bin/env bash
set -euo pipefail

# Bootstrap GPU artifact directory with prerequisite files from local validation run
# Usage: bash scripts/gpu_bootstrap_prerequisites.sh

SOURCE_BASE="artifacts/final_cost_ladder/04"
TARGET_BASE="artifacts/kestrel_gpu_h100_run"

echo "Bootstrapping GPU artifacts from local validation run..."

# Create target directories
mkdir -p "$TARGET_BASE/output_conditioning"
mkdir -p "$TARGET_BASE/empirical_null_screen"
mkdir -p "$TARGET_BASE/interaction_discovery"

# Copy output_conditioning prerequisites
echo "Copying output_conditioning artifacts..."
cp "$SOURCE_BASE/output_conditioning/pca_scores.csv" "$TARGET_BASE/output_conditioning/"
cp "$SOURCE_BASE/output_conditioning/pca_loadings.csv" "$TARGET_BASE/output_conditioning/"
cp "$SOURCE_BASE/output_conditioning/pca_explained_variance.csv" "$TARGET_BASE/output_conditioning/"

# Copy empirical_null_screen prerequisites
echo "Copying empirical_null_screen artifacts..."
cp "$SOURCE_BASE/empirical_null_screen/retained_terms.csv" "$TARGET_BASE/empirical_null_screen/"
cp "$SOURCE_BASE/empirical_null_screen/component_coefficients.csv" "$TARGET_BASE/empirical_null_screen/"

# Copy interaction_discovery inputs (X, holdout_assignments, feature_catalog)
# These are typically at artifact_dir root, not in a subdirectory
echo "Copying interaction_discovery inputs..."
cp "artifacts/test_dataset_3k/X.parquet" "$TARGET_BASE/" || true
cp "artifacts/test_dataset_3k/holdout_assignments.parquet" "$TARGET_BASE/" || true
cp "artifacts/test_dataset_3k/actual_input_feature_catalog.parquet" "$TARGET_BASE/" || true

echo "Bootstrap complete. GPU artifacts ready in $TARGET_BASE"
ls -lh "$TARGET_BASE/output_conditioning/"
ls -lh "$TARGET_BASE/empirical_null_screen/"
