#!/bin/bash
# Publication-ready full-dataset study: Step 4 — Collect artifacts
#
# Purpose: Pull final results from HPC to local machine
# Output: Complete study package with tables, figures, metadata, reproducibility
# Time: ~5-10 minutes
#
# Usage: bash scripts/publication-run/04_collect_publication_artifacts.sh

set -euo pipefail

echo "RETIRED: use the content-addressed G11 campaign package; this legacy workflow is non-executable." >&2
exit 64

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================================"
echo "Publication-Ready Full-Dataset Study: COLLECT FINAL ARTIFACTS"
echo "================================================================================"
echo ""
echo "Orchestration config: configs/hpc/kestrel_publication_orchestration.yml"
echo "Mode: reporting_bundle (final tables, figures, metadata)"
echo ""
echo "⏳ Collecting artifacts from HPC..."
echo ""

# Run collection
pixi run hpc-workflow -- \
  --config configs/hpc/kestrel_publication_orchestration.yml \
  --action collect

echo ""
echo "================================================================================"
echo "✓ COLLECTION COMPLETE"
echo "================================================================================"
echo ""

# Find the most recent bundle
LATEST_BUNDLE=$(ls -t ./artifacts/kestrel_collected_bundles/kestrel_hpc_snapshot_*.zip 2>/dev/null | head -1 || true)
if [ -z "$LATEST_BUNDLE" ]; then
  echo "⚠️  No bundle found. Check collection output above for errors."
  exit 1
fi

BUNDLE_SIZE=$(du -h "$LATEST_BUNDLE" | cut -f1)
BUNDLE_NAME=$(basename "$LATEST_BUNDLE")

echo "Bundle location: ./artifacts/kestrel_collected_bundles/$BUNDLE_NAME"
echo "Bundle size: $BUNDLE_SIZE"
echo ""

# Extract to working directory
EXTRACT_DIR="./artifacts/publication_full_dataset_results"
rm -rf "$EXTRACT_DIR" && mkdir -p "$EXTRACT_DIR"
unzip -q "$LATEST_BUNDLE" -d "$EXTRACT_DIR"

echo "Extracted to: $EXTRACT_DIR"
echo ""

# Check for expected artifacts
echo "Verifying publication-grade artifacts..."
echo ""

CHECKS=(
  "manifest/hpc_run_manifest.json"
  "manifest/run_summary.csv"
  "manifest/commands.json"
  "manifest/reproduction_recipe.md"
  "manifest/study_metadata_manifest.json"
)

# Per-stage canonical artifacts that must be present for the manuscript
# tables/figures to be reproducible. These are the post-reduce artifact
# directory names (writer convention from
# rfm_pipeline.manuscript_pipeline_helpers); the HPC stage names used
# in --stage / hpc_shards_<stage> / SLURM job names differ for
# empirical_null_screening (the HPC name carries -ing; the writer dir
# does not).
STAGE_DIR_CHECKS=(
  "output_conditioning"
  "empirical_null_screen"
  "interaction_discovery"
  "nonlinear_discovery"
  "sparse_selection"
  "final_manuscript_artifacts"
)

ALL_PRESENT=true
for check_path in "${CHECKS[@]}"; do
  full_path="$EXTRACT_DIR/$check_path"
  if [ -f "$full_path" ]; then
    SIZE=$(du -h "$full_path" | cut -f1)
    echo "  ✓ $check_path ($SIZE)"
  else
    echo "  ✗ MISSING: $check_path"
    ALL_PRESENT=false
  fi
done

# Verify that at least one stage artifact directory exists per stage in
# at least one run target. Missing stage artifacts mean the manuscript
# tables/figures cannot be regenerated even if the manifest is intact.
if [ -d "$EXTRACT_DIR/runs" ]; then
  echo ""
  echo "Stage artifact directories (must exist for manuscript reproducibility):"
  for stage in "${STAGE_DIR_CHECKS[@]}"; do
    # Restrict to canonical layout: runs/<target>/<stage>/ (depth 2 from
    # runs/). Looser `-name "$stage"` would also match nested SLURM
    # script dirs and shard merge dirs under hpc_shards_<stage>/.
    if find "$EXTRACT_DIR/runs" -mindepth 2 -maxdepth 2 -type d -name "$stage" -print -quit | grep -q .; then
      echo "  ✓ $stage"
    else
      echo "  ✗ MISSING: $stage (no run target contains this stage's artifacts)"
      ALL_PRESENT=false
    fi
  done
else
  # A bundle that contains a manifest but no runs/ directory cannot
  # regenerate any manuscript table or figure. Flag this loudly rather
  # than letting the script print "All critical artifacts present"
  # solely on the strength of the manifest files above.
  echo ""
  echo "  ✗ MISSING: runs/ (no per-target stage artifact directories in bundle)"
  ALL_PRESENT=false
fi

echo ""
echo "================================================================================"
echo "ARTIFACT COLLECTION STATUS"
echo "================================================================================"
echo ""

if [ "$ALL_PRESENT" = true ]; then
  echo "✓ All critical artifacts present and ready for manuscript"
  echo ""
  echo "Study package contains:"
  echo "  • Reproducibility manifest (commands, configs, versions)"
  echo "  • Stage-by-stage execution summary"
  echo "  • Raw results (tables, interaction networks, coefficients)"
  echo "  • Publication figures (SVG format)"
  echo "  • Statistical summaries (bootstrap confidence intervals)"
  echo ""
  echo "Location: ./artifacts/publication_full_dataset_results/"
  echo ""
  echo "Next steps:"
  echo "  1. Review manifest: cat $EXTRACT_DIR/manifest/study_metadata_manifest.json"
  echo "  2. Check figures: ls -lh $EXTRACT_DIR/*/figures/"
  echo "  3. Review summary: cat $EXTRACT_DIR/manifest/run_summary.csv"
  echo ""
  echo "Ready for manuscript integration!"
else
  echo "⚠️  Some artifacts missing. Review the bundle contents."
  echo "    Check: ls -R $EXTRACT_DIR"
  echo ""
  # Exit non-zero so the master orchestrator (set -euo pipefail) halts
  # before printing a misleading "WORKFLOW COMPLETE" banner. Operators
  # can rerun collection after diagnosing the missing artifacts.
  exit 1
fi

echo ""
