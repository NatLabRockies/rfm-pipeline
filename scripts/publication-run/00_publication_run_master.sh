#!/bin/bash
# Publication-ready full-dataset study: MASTER ORCHESTRATOR
#
# Purpose: Execute complete workflow with user prompts between steps
# Stages: Dry-run → Live submission → Monitor → Collect
#
# Usage: bash scripts/publication-run/00_publication_run_master.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_step() {
  echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${GREEN}STEP: $1${NC}"
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

prompt_continue() {
  echo -e "\n${YELLOW}Continue to next step? (yes/no)${NC}"
  read -r response
  if [ "$response" != "yes" ]; then
    echo "Exiting."
    exit 0
  fi
}

# Main workflow
echo -e "${GREEN}"
cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║  Publication-Ready Full-Dataset Study — Master Orchestrator   ║
║                                                               ║
║  Dataset: real BSM full data (30,000 samples)                ║
║  Config: kestrel_publication_full_dataset.yml                ║
║  Target: Quality-first analysis with reproducible results    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo "This orchestrator will guide you through all 4 stages:"
echo "  1. Dry-run validation (scripts generated but NOT submitted)"
echo "  2. Live HPC submission (submit job to queue)"
echo "  3. Status monitoring (track progress through all stages)"
echo "  4. Artifact collection (pull results from HPC)"
echo ""
echo "Estimated total time:"
echo "  • Step 1 (dry-run):     ~30 seconds"
echo "  • Step 2 (submit):      ~2 minutes"
echo "  • Step 3 (monitor):     ~24 hours (manual polling)"
echo "  • Step 4 (collect):     ~10 minutes"
echo ""

# Step 1: Dry-run
log_step "DRY-RUN VALIDATION"
echo "Generating and validating SLURM scripts without submission..."
echo ""
bash scripts/publication-run/01_dry_run_submission.sh
prompt_continue

# Step 2: Live submission
log_step "LIVE SUBMISSION"
echo "Submitting publication-grade job to HPC..."
echo ""
bash scripts/publication-run/02_live_submission.sh
prompt_continue

# Step 3: Monitoring (with user loop)
log_step "EXECUTION MONITORING"
echo "Starting HPC status monitoring..."
echo ""
echo "You can:"
echo "  • Monitor manually: bash scripts/publication-run/03_monitor_publication_run.sh"
echo "  • Run automated checks with interval: bash scripts/publication-run/03_monitor_publication_run.sh 300"
echo "  • Continue to collection when all stages are complete"
echo ""

MONITOR=true
while [ "$MONITOR" = true ]; do
  echo -e "${YELLOW}Run monitoring check now? (yes/no/skip-to-collect)${NC}"
  read -r response
  case "$response" in
    yes)
      bash scripts/publication-run/03_monitor_publication_run.sh
      echo ""
      ;;
    no)
      echo "⏳ You can run monitoring later: bash scripts/publication-run/03_monitor_publication_run.sh"
      MONITOR=false
      ;;
    skip-to-collect)
      MONITOR=false
      ;;
    *)
      echo "Invalid response. Please enter 'yes', 'no', or 'skip-to-collect'"
      ;;
  esac
done

# Step 4: Collection
log_step "ARTIFACT COLLECTION"
echo "Pulling final results from HPC..."
echo ""
bash scripts/publication-run/04_collect_publication_artifacts.sh

echo -e "\n${GREEN}"
cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║                    ✓ WORKFLOW COMPLETE                        ║
║                                                               ║
║  Publication-ready results available at:                     ║
║  ./artifacts/publication_full_dataset_results/               ║
║                                                               ║
║  Next: Review manifest and prepare for manuscript            ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"
