#!/usr/bin/env bash
# =============================================================================
# BSM HPC Single-Node Scaling Benchmark
# =============================================================================
#
# PURPOSE
#   Test the BSM manuscript pipeline's multicore performance on a single HPC
#   node and build a timing dataset for projecting large-run runtimes.
#
# USAGE (copy entire repo or just this script + pixi.toml to a node)
#
#   # Interactive node (Kestrel example):
#   salloc --account=bsm --partition=shared --nodes=1 --ntasks=1 \
#          --cpus-per-task=104 --mem=240G --time=08:00:00
#   bash scripts/hpc_node_benchmark.sh
#
#   # Or as a batch job:
#   sbatch scripts/hpc_node_benchmark.sh
#
#   # Quick smoke test (3 cells, ~5 minutes):
#   bash scripts/hpc_node_benchmark.sh --quick
#
#   # Specify output directory:
#   bash scripts/hpc_node_benchmark.sh \
#       --output-root /scratch/$USER/bsm_benchmark \
#       --dataset-path /projects/bsm/bsm-public-rf/artifacts/test_dataset_3k
#
# SBATCH HEADER (uncomment to use as a batch script)
# --------------------------------------------------------------------------
#SBATCH --job-name=bsm_scaling_benchmark
#SBATCH --account=bsm
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=bsm_benchmark_%j.out
#SBATCH --error=bsm_benchmark_%j.err
# --------------------------------------------------------------------------
#
# WHAT IT DOES
#   1. Detects CPU count and activates the Pixi environment
#   2. Runs hpc_scaling_benchmark.py to sweep key configuration axes:
#      - Core scaling     : n_jobs ∈ {1, 2, 4, 8, 16, 32, 64, max}
#      - Feature scaling  : max_retained_terms ∈ {20, 40, 75, 100, 150, 200, 300, uncapped}
#      - Permutation count: n_permutations ∈ {5, 11, 21, 41, 101}
#      - Tree count       : n_tree_estimators ∈ {25, 50, 100, 200}
#      - 2D cross sweep   : selected (n_jobs × max_retained_terms) pairs
#   3. Runs hpc_compute_calculator.py to fit the scaling model and produce:
#      - A human-readable report with decision tables
#      - A JSON model file for downstream use
#   4. Prints a summary and tells you where the results are
#
# OUTPUT FILES
#   <output_root>/scaling_results.csv        -- per-cell timing data
#   <output_root>/benchmark_manifest.json   -- run metadata
#   <output_root>/benchmark_log.txt         -- per-cell stdout/stderr
#   <output_root>/compute_model.json        -- fitted scaling model
#   <output_root>/compute_report.md         -- human-readable report + tables
#
# NEXT STEPS
#   After the benchmark, share scaling_results.csv with collaborators or
#   run the calculator interactively:
#
#   pixi run hpc-compute-calculator -- \
#       --results <output_root>/scaling_results.csv \
#       --n-jobs <your_cores> \
#       --max-retained-terms <your_feature_count> \
#       --n-samples <your_sample_count>
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
QUICK=0
OUTPUT_ROOT=""
DATASET_PATH=""
MAX_CORES=""
SWEEPS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)        QUICK=1; shift ;;
        --output-root)  OUTPUT_ROOT="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --max-cores)    MAX_CORES="$2"; shift 2 ;;
        --sweeps)       SWEEPS="$2"; shift 2 ;;
        --help|-h)
            head -70 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "BSM HPC Scaling Benchmark"
echo "  Repo  : $REPO_ROOT"
echo "  Node  : $(hostname)"
echo "  Date  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# Detect cores
if [[ -z "$MAX_CORES" ]]; then
    MAX_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "1")
fi
echo "  Cores : $MAX_CORES available"

# Default output root
if [[ -z "$OUTPUT_ROOT" ]]; then
    SCRATCH="${SCRATCH:-/scratch/${USER:-benchmark}}"
    OUTPUT_ROOT="${SCRATCH}/bsm_scaling_benchmark_$(date +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$OUTPUT_ROOT"
echo "  Output: $OUTPUT_ROOT"

# ---------------------------------------------------------------------------
# Activate Pixi environment
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Activating Pixi environment..."

# Try several common Pixi locations
PIXI_BIN=""
for candidate in \
    "${REPO_ROOT}/.pixi/envs/default/bin/python" \
    "/projects/bsm/.pixi/envs/default/bin/python" \
    "${HOME}/.pixi/envs/default/bin/python"; do
    if [[ -f "$candidate" ]]; then
        PIXI_BIN="$(dirname "$candidate")"
        break
    fi
done

# If pixi is on PATH, use it directly
if command -v pixi &>/dev/null; then
    echo "  Found pixi on PATH: $(which pixi)"
    PIXI_CMD="pixi run"
    PYTHON_CMD="pixi run python"
else
    if [[ -z "$PIXI_BIN" ]]; then
        echo "ERROR: Could not find pixi or Pixi environment." >&2
        echo "  Run: curl -fsSL https://pixi.sh/install.sh | bash" >&2
        echo "  Then: cd $REPO_ROOT && pixi install --locked" >&2
        exit 1
    fi
    echo "  Found Pixi env at: $PIXI_BIN"
    export PATH="$PIXI_BIN:$PATH"
    PIXI_CMD="pixi run"
    PYTHON_CMD="python"
fi

# Verify environment works
cd "$REPO_ROOT"
$PYTHON_CMD -c "import bsm_rfm; print('  bsm_rfm OK:', bsm_rfm.__file__)"

# ---------------------------------------------------------------------------
# Build benchmark command
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Running scaling benchmark (~2–8 hours for full grid)..."
echo ""

BENCH_ARGS=(
    "--output-root" "$OUTPUT_ROOT"
    "--max-cores" "$MAX_CORES"
)

if [[ -n "$DATASET_PATH" ]]; then
    BENCH_ARGS+=("--dataset-path" "$DATASET_PATH")
fi

if [[ $QUICK -eq 1 ]]; then
    echo "  Quick mode: 3-cell smoke grid only"
    BENCH_ARGS+=("--quick")
fi

if [[ -n "$SWEEPS" ]]; then
    BENCH_ARGS+=("--sweeps" $SWEEPS)
fi

# Run benchmark (checkpoints after every cell)
$PYTHON_CMD tools/hpc_scaling_benchmark.py "${BENCH_ARGS[@]}"

RESULTS_CSV="${OUTPUT_ROOT}/scaling_results.csv"
if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "ERROR: benchmark produced no results CSV at $RESULTS_CSV" >&2
    exit 1
fi

N_ROWS=$(tail -n +2 "$RESULTS_CSV" | grep -c "^" || echo 0)
echo ""
echo "  Benchmark complete: $N_ROWS successful cells"

# ---------------------------------------------------------------------------
# Fit scaling model and generate report
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Fitting scaling model and generating compute report..."

MODEL_JSON="${OUTPUT_ROOT}/compute_model.json"
REPORT_MD="${OUTPUT_ROOT}/compute_report.md"

$PYTHON_CMD tools/hpc_compute_calculator.py \
    --results "$RESULTS_CSV" \
    --max-cores "$MAX_CORES" \
    --sweep-cores 1 2 4 8 16 32 64 "$MAX_CORES" \
    --save-model "$MODEL_JSON" \
    --report "$REPORT_MD" || {
    echo "WARNING: Calculator failed (need ≥3 cells). Skipping." >&2
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[4/4] Benchmark Complete"
echo "============================================================"
echo ""
echo "  Results CSV  : $RESULTS_CSV"
echo "  Scaling model: $MODEL_JSON"
echo "  Report       : $REPORT_MD"
echo ""
echo "To estimate your full run:"
echo ""
echo "  pixi run hpc-compute-calculator -- \\"
echo "      --results $RESULTS_CSV \\"
echo "      --n-jobs $MAX_CORES \\"
echo "      --max-retained-terms 350 \\"
echo "      --n-permutations 21 \\"
echo "      --n-tree-estimators 100 \\"
echo "      --n-samples 30000"
echo ""
echo "To share results: copy $RESULTS_CSV to your local machine."
echo ""
