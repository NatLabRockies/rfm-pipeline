# Documentation Improvement Plan

**Status:** Created 2026-05-08\
**Goal:** Make documentation complete, organized, and easy to use for new users

## Current State Assessment

### ✅ Strengths

- Comprehensive technical documentation exists
- 9 well-documented manuscript notebooks
- Working example script (end_to_end_reproducibility.py)
- Config-driven workflow is functional
- Good coverage of scientific details

### ❌ Gaps

- No interactive quick-start notebook (users learn by doing)
- Artifact reference incomplete (only covers stage 1 of 6)
- No results interpretation guide (how to know if output is good?)
- Multiple redundant quick-start docs without clear hierarchy
- No GitHub Pages site (hard to browse, no search)
- Missing troubleshooting guide
- Config dataset docs are minimal

## Improvement Tasks

### Priority 1: Core User Experience (Do First)

#### 1. Complete Artifact Reference

**File:** `docs/artifact_reference.md`\
**Status:** ⏳ In progress (only stage 1 done)\
**Action:** Document all 6 stages:

- Stage 1: Output Conditioning ✅
- Stage 2: Empirical Null Screening ❌
- Stage 3: Interaction Discovery ❌
- Stage 4: Nonlinear Discovery ❌
- Stage 5: Sparse Selection ❌
- Stage 6: Final Artifacts & Export ❌

For each stage, document:

- All output files (CSV, parquet, figures)
- Column schemas with types
- Expected value ranges
- Interpretation guidance
- What to check for success/failure

#### 2. Create Results Interpretation Guide

**File:** `docs/interpreting_results.md` (new)\
**Action:** Write comprehensive guide covering:

- How to validate your run succeeded
- Benchmark values from manuscript:
  - Output conditioning: ~39 PCs, 90% variance
  - Empirical null: ~349 retained features at BH q=0.10
  - Interaction discovery: ~367 retained pairs
  - Nonlinear discovery: ~112 transformations
  - Sparse selection: ~340 features per component
  - Final model: nRMSE ~0.15-0.25 range
- Red flags (run likely failed)
- How to compare your results to manuscript
- What constitutes "good" results for custom data

#### 3. Enhance Dataset Configuration Docs

**File:** `configs/datasets/README.md`\
**Action:** Expand from bare-bones to comprehensive:

- Step-by-step walkthrough with screenshots
- Multiple worked examples:
  - Synthetic data config
  - Test dataset (3k samples)
  - Real data (20k samples)
  - Custom user data
- Data format requirements (sample_id column, scenario factors)
- Preprocessing guidance
- Common pitfalls and how to avoid them
- Config validation checklist
- Troubleshooting section

#### 4. Consolidate Quick-Start Documentation

**Action:** Merge/organize redundant docs:

- **Keep:** `README.md` (primary entry point)
- **Keep:** `docs/setup_and_first_run.md` (detailed setup)
- **Keep:** `docs/quickstart.md` (API examples)
- **Archive:** `COMPLETE_SETUP_GUIDE.md` → merge into setup_and_first_run.md
- **Archive:** `MANUSCRIPT_QUICK_START.md` → merge into RUNNING_MANUSCRIPT_REPRODUCTION.md
- **Remove:** `SIMPLE_CONFIG_SUMMARY.md` (working doc, superseded)
- **Remove:** `WORKFLOW_SUMMARY.md` (working doc, superseded)

Create clear documentation path:

1. First time? → `README.md` quick start
1. Need details? → `docs/setup_and_first_run.md`
1. API usage? → `docs/quickstart.md`
1. Config workflow? → `docs/RUNNING_MANUSCRIPT_REPRODUCTION.md`

### Priority 2: Interactive Learning

#### 5. Create Quick-Start Notebook

**File:** `notebooks/quickstart.ipynb` (new)\
**Dependencies:** Enhanced dataset config docs\
**Action:** Build 10-minute interactive walkthrough:

**Section 1: Setup (2 min)**

- Load pre-built test dataset (3k samples)
- Verify data format
- Examine config file

**Section 2: Run Workflow (3 min)**

- Execute config-driven workflow
- Show progress through 6 stages
- Point out key intermediate outputs

**Section 3: Examine Results (3 min)**

- Load final artifacts
- Visualize feature selection across stages
- Plot holdout performance
- Show example predictions

**Section 4: Interpret (2 min)**

- Compare to manuscript benchmarks
- Validate success
- Next steps (try your own data)

**Requirements:**

- Self-contained (uses included test data)
- Executable in Binder/Colab
- Clear markdown explanations
- Plots render inline
- Takes \<5 minutes to run

### Priority 3: Discoverability & Organization

#### 6. Set Up GitHub Pages

**Dependencies:** Consolidated docs, complete artifact reference\
**Action:** Deploy mkdocs-material site at `natlabrockies.github.io/bsm-public-rf`

**Structure:**

```
Home
├── Getting Started
│   ├── Quick Start (5-min demo)
│   ├── Installation
│   ├── First Run
│   └── Quick-Start Notebook
├── User Guide
│   ├── Config-Driven Workflow
│   ├── Creating Datasets
│   ├── Interpreting Results
│   └── Troubleshooting
├── Reference
│   ├── Artifact Reference
│   ├── Configuration Reference
│   ├── API Reference (auto-generated)
│   └── Workflow Stages
├── Advanced
│   ├── Manuscript Reproduction
│   ├── Custom Datasets
│   └── Development Guide
└── Notebooks
    ├── Quick Start
    └── Manuscript Stages (01-08)
```

**Setup:**

- Install mkdocs-material in pixi environment
- Create `.github/workflows/docs.yml` for auto-deployment
- Add search functionality
- Mobile-responsive design
- Version selector (future feature)

**Files to create:**

- `mkdocs.yml` - Site configuration
- `docs/index.md` - Landing page with clear navigation
- `.github/workflows/docs.yml` - CI deployment

#### 7. Add Troubleshooting Guide

**File:** `docs/troubleshooting.md` (new)\
**Action:** Document common issues:

**Installation Issues**

- Pixi not found → add to PATH
- Lock file conflicts → pixi install --locked
- Missing dependencies → specific package fixes

**Data Issues**

- Missing sample_id column → use preprocessing script
- MultiIndex format → use preprocess_multiindex_data.py
- Wrong split values (train/test vs train/holdout)
- Scenario factors (AFSC, UAEORO) missing

**Runtime Issues**

- Out of memory → reduce candidate library size
- Process hanging → check feature catalog size
- Permission errors → check output directory
- Import errors → PYTHONPATH=src

**Results Issues**

- All features filtered → check variance threshold
- No interactions found → verify catalog format (colon-delimited)
- Poor holdout performance → check data quality, sample size

**FAQ Section:**

- How long should it take? (3k: ~20 min, 20k: ~2 hrs)
- How much memory needed? (8GB min, 16GB recommended)
- Can I use my own data? (Yes, see Creating Datasets guide)
- What if results don't match manuscript? (Expected, see benchmarks)

## Implementation Notes

### Execution Order

1. **Complete artifact reference** (no dependencies)
1. **Enhance dataset config docs** (no dependencies)
1. **Create results interpretation guide** (no dependencies)
1. **Add troubleshooting guide** (no dependencies)
1. **Consolidate quick-start docs** (depends on 1-3)
1. **Create quick-start notebook** (depends on 2)
1. **Set up GitHub Pages** (depends on 5, 1)

### Documentation Standards

- Use MyST Markdown (compatible with Sphinx and mkdocs)
- NumPy-style docstrings for all public APIs
- Include examples in every guide
- Add "Prerequisites" and "Next Steps" to each doc
- Use admonitions for warnings/tips/notes
- Link between related docs
- Keep line length \<100 chars for readability

### Testing Documentation

- All code examples must be tested
- Notebooks must execute without errors
- Links must be validated (no 404s)
- Install fresh environment and follow guides literally
- Test on both macOS and Linux if possible

## Success Metrics

Documentation improvement is successful when:

- ✅ New user can run workflow in \<15 minutes
- ✅ All artifacts have clear interpretation guidance
- ✅ Quick-start notebook executes without errors
- ✅ GitHub Pages site is searchable and mobile-friendly
- ✅ Zero redundant/conflicting quick-start docs
- ✅ Common errors have documented solutions
- ✅ User can validate results without asking questions

## Future Enhancements (Not Urgent)

- Video tutorial (5-minute walkthrough)
- Contributing guide for developers
- API reference auto-generation from docstrings
- Example gallery (use cases beyond manuscript)
- Integration with Binder for zero-install demos
- Version selector for different releases
- Jupyter Book integration for richer notebooks
- Community forum or discussion board
