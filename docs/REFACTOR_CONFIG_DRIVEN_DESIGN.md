# Config-Driven Unified Workflow Refactor: Design Document

**Milestone**: Phase 5 (queued, starts after 300-sample parallel run validates)
**Status**: Planning phase (do not implement yet)
**Priority**: High (eliminates script duplication, enables multiple run modes)
**Estimated effort**: 6-8 hours

______________________________________________________________________

## Problem Statement

Currently, validation and execution logic is scattered across dataset-specific scripts:

- `tools/run_300_sample_validation.py` — hardcoded 300-sample dataset, with `--no-caps` flag for parallel/serial toggle
- No unified way to run manuscript pipeline on arbitrary datasets
- No config schema for parameterizing: dataset, caps, parallelization, algorithm params
- Difficult to add new run modes (CI fast, full-dataset, custom) without duplicating code

**Design goal**: Single entry point + config files → eliminates hardcoding, enables rapid new modes

______________________________________________________________________

## Solution Architecture

### Layer 1: Config Schema (Unified)

**File**: `docs/CONFIGURATION_REFERENCE.md` (new) + `configs/*.yml` (new directory)

**Config structure**:

```yaml
# configs/validation_300_sample_no_caps.yml
dataset:
  type: "synthetic_300_sample"          # ID to locate dataset
  path: "./data/synthetic_300_sample/"  # Or derived from type

algorithm:
  variance_threshold: 0.90
  retained_components: null             # If null, derive from variance_threshold

runtime:
  n_jobs: -1                            # -1 = all CPUs, 1 = serial
  output_batch_size: null               # For final OLS chunking

stages:
  empirical_null_screening:
    n_permutations: 1000                # B+1 permutations
    bh_q_threshold: 0.10

  interaction_discovery:
    p_threshold: 0.05
    n_tree_estimators: 100              # SHAP tree BARF
    max_tree_depth: 10

  nonlinear_discovery:
    edf_threshold: 2.5
    transform_library:  # list of {expr, label, name}

  sparse_selection:
    n_stability_subsamples: 100
    subsample_fraction: 1.0
    lasso_alpha_grid_size: 40

  final_artifacts:
    bootstrap_count: 100
    bootstrap_alpha: 0.95

validation:
  # Optional fast-mode overrides (for CI)
  fast_mode: false
  fast_mode_overrides:
    n_permutations: 5
    n_tree_estimators: 20
    n_stability_subsamples: 8
    bootstrap_count: 20
    output_cap: 300

output:
  artifact_dir: "./artifacts/validation_300_sample_no_caps/"
  seed: 0
  verbose: true
```

### Layer 2: Config Parser (Typed)

**File**: `src/rfm_pipeline/config_parser.py` (new)

```python
from dataclasses import dataclass
from typing import Optional, Dict, List
import yaml

@dataclass
class DatasetConfig:
    type: str
    path: str

@dataclass
class AlgorithmConfig:
    variance_threshold: float
    retained_components: Optional[int] = None

@dataclass
class RuntimeConfig:
    n_jobs: int = 1
    output_batch_size: Optional[int] = None

@dataclass
class StageConfig:
    empirical_null_screening: Dict
    interaction_discovery: Dict
    nonlinear_discovery: Dict
    sparse_selection: Dict
    final_artifacts: Dict

@dataclass
class WorkflowConfig:
    dataset: DatasetConfig
    algorithm: AlgorithmConfig
    runtime: RuntimeConfig
    stages: StageConfig
    validation: Dict
    output: Dict

def load_config(path: str) -> WorkflowConfig:
    """Load and validate config YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    # Construct WorkflowConfig from data with validation
    return WorkflowConfig(...)

def apply_fast_mode_overrides(config: WorkflowConfig) -> WorkflowConfig:
    """Apply fast-mode overrides if enabled."""
    if config.validation.get("fast_mode"):
        overrides = config.validation.get("fast_mode_overrides", {})
        # Apply overrides to config in-place
        config.stages.empirical_null_screening["n_permutations"] = overrides.get("n_permutations", ...)
        # ... etc
    return config
```

### Layer 3: Unified Runner (Entry Point)

**File**: `tools/run_manuscript_pipeline.py` (new, replaces run_300_sample_validation.py)

```python
#!/usr/bin/env python
"""Unified manuscript pipeline runner with config-driven parameterization."""

import argparse
import yaml
from pathlib import Path

from src.rfm_pipeline.config_parser import load_config, apply_fast_mode_overrides
from src.rfm_pipeline.manuscript_stages import run_manuscript_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Run manuscript pipeline with config file"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to config YAML file (e.g., configs/validation_300_sample_no_caps.yml)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast-mode overrides (useful for CI/testing)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output artifact directory"
    )
    args = parser.parse_args()

    # Load and validate config
    config = load_config(args.config)

    # Apply fast-mode overrides if requested
    if args.fast:
        config.validation["fast_mode"] = True
        config = apply_fast_mode_overrides(config)

    # Override with CLI args if provided
    if args.seed is not None:
        config.output["seed"] = args.seed
    if args.output_dir is not None:
        config.output["artifact_dir"] = args.output_dir

    # Run pipeline with config
    results = run_manuscript_pipeline(config)

    print(f"Pipeline complete. Artifacts: {config.output['artifact_dir']}")
    return 0

if __name__ == "__main__":
    exit(main())
```

**Usage**:

```bash
# Full run with all parameters
pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_no_caps.yml

# Fast mode (CI override)
pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_full.yml --fast

# Custom output directory
pixi run python tools/run_manuscript_pipeline.py configs/full_manuscript.yml --output-dir /tmp/custom_run

# Custom seed
pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_no_caps.yml --seed 42
```

### Layer 4: Configuration Examples

**Directory**: `configs/` (new)

```
configs/
├── validation_300_sample_fast.yml        # For unit tests + CI (fast)
├── validation_300_sample_no_caps.yml     # Full 300-sample run (parallel)
├── validation_300_sample_serial.yml      # Full 300-sample run (serial)
├── full_manuscript.yml                   # Full dataset (if available)
├── _defaults.yml                         # Shared defaults (via YAML anchors)
└── README.md                              # Config guide
```

**Inheritance pattern** (YAML anchors):

```yaml
# configs/_defaults.yml
defaults: &defaults
  variance_threshold: 0.90
  p_threshold: 0.05
  edf_threshold: 2.5

# configs/validation_300_sample_fast.yml
<<: *defaults
dataset:
  type: synthetic_300_sample
runtime:
  n_jobs: 1
validation:
  fast_mode: true
```

______________________________________________________________________

## Integration with manuscript_stages.py

**Key change**: `run_manuscript_pipeline()` signature updated to accept config:

```python
# Before
def run_manuscript_pipeline(case_study_config: dict, caps_dict: Optional[dict] = None):
    ...

# After
def run_manuscript_pipeline(workflow_config: WorkflowConfig):
    # Extract legacy case_study_config from workflow_config
    case_study_config = convert_workflow_config_to_legacy_case_study(workflow_config)

    # Wire n_jobs from runtime config
    n_jobs = workflow_config.runtime.n_jobs

    # Call existing pipeline logic
    results = _run_pipeline_stages(case_study_config, n_jobs=n_jobs)
    return results
```

**Backward compatibility**:

- `run_300_sample_validation.py` still exists but now calls `run_manuscript_pipeline.py` internally
- All existing tests continue to pass (use legacy case_study_config directly)
- New tests added for config parsing + validation

______________________________________________________________________

## Migration Checklist

### Step 1: Create config_parser.py

- Define WorkflowConfig and sub-dataclasses
- Implement load_config() with YAML parsing and validation
- Implement apply_fast_mode_overrides()
- Add unit tests for config loading

### Step 2: Create configs/ directory with examples

- \_defaults.yml (common params)
- validation_300_sample_fast.yml (CI mode)
- validation_300_sample_no_caps.yml (full 300-sample)
- validation_300_sample_serial.yml (baseline serial)
- full_manuscript.yml (full dataset, if available)

### Step 3: Create run_manuscript_pipeline.py

- Parse CLI args
- Load config
- Integrate with manuscript_stages.py
- Return results with metadata

### Step 4: Update run_300_sample_validation.py

- Import run_manuscript_pipeline
- Parse CLI args (`--no-caps`, `--output-root`, etc.)
- Map to config file selection + overrides
- Call run_manuscript_pipeline internally

### Step 5: Add tests

- `tests/test_config_parser.py` — config loading + validation
- `tests/test_run_manuscript_pipeline.py` — runner integration
- `tests/test_cli_args.py` — CLI argument parsing

### Step 6: Update docs

- `docs/CONFIGURATION_REFERENCE.md` — config schema + examples
- `docs/MANUSCRIPT_WORKFLOW_REFERENCE.md` — add section on new entry point
- `docs/AGENT_SYNC.md` — record completion

### Step 7: Full validation

- `bash ./test_repo.sh --check` — all tests pass
- `pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_no_caps.yml` — produces same artifacts as before
- `pixi run python tools/run_300_sample_validation.py --no-caps` — backward compatible

### Step 8: Commit

- Commit with message: `refactor: config-driven unified pipeline entry point; replace script-specific hardcoding`
- Push to main

______________________________________________________________________

## Non-Goals & Constraints

- **No config hotloading**: Config loaded once at startup; changes require script restart
- **No dynamic algorithm selection**: All 6 stages always run (can skip via config flags in future)
- **No external config validation library**: Use dataclass validation only
- **No config merging from multiple files**: Single config per run (can use YAML anchors for DRY)
- **No environment variable substitution**: Explicit config values only

______________________________________________________________________

## Backward Compatibility

**What must NOT change**:

- `manuscript_stages.py` API (specs, run_manuscript_pipeline, etc.)
- `final_ols.py` API
- Test file structure
- Artifact format/location

**What SHOULD change**:

- Remove hardcoding from validator scripts
- Add config schema + parsing layer
- Create single entry point

**How to verify**:

- All existing tests pass without modification
- `run_300_sample_validation.py --no-caps` produces identical artifacts to before
- Full gate passes

______________________________________________________________________

## Timeline

**Estimated effort**: 6-8 hours

| Task                                | Time |
| ----------------------------------- | ---- |
| config_parser.py + unit tests       | 1.5h |
| Create configs/ + examples          | 1h   |
| run_manuscript_pipeline.py          | 1.5h |
| Update run_300_sample_validation.py | 1h   |
| Add integration tests               | 1.5h |
| Update docs                         | 1h   |
| Full validation + commit            | 1h   |

**Start**: Only after 300-sample parallel run validates
**Blocking**: None (queue behind validation)

______________________________________________________________________

## Success Criteria

- [x] Waiting for 300-sample validation
- [ ] `run_manuscript_pipeline.py` accepts config YAML
- [ ] `configs/` directory with 3+ example configs
- [ ] `docs/CONFIGURATION_REFERENCE.md` documents schema
- [ ] All 7 refactor todos marked done
- [ ] Full gate passes
- [ ] Backward compatibility verified (artifacts identical)
- [ ] Commit on main with descriptive message

______________________________________________________________________

## Risks & Mitigations

| Risk                     | Mitigation                                       |
| ------------------------ | ------------------------------------------------ |
| Config parsing bugs      | Unit tests for each config load path             |
| Backward incompatibility | Run both old + new scripts, compare artifacts    |
| Missing edge cases       | Cover all manuscript parameters in config schema |
| YAML syntax errors       | Add schema validation + clear error messages     |

______________________________________________________________________

## Future Extensions

Once refactor complete, these become easy:

1. **Add GPU solver for sparse selection**: New config field `sparse_selection.solver: "gpu"` + loader logic
1. **Add mixed-effect model support**: New transform family in nonlinear discovery
1. **Add bootstrap resampling**: New config field + stage
1. **CI integration**: Use fast-mode config automatically in GitHub Actions
1. **Multi-dataset benchmark suite**: Directory of configs; batch runner script

______________________________________________________________________

## References

- Config schema: Pydantic (future) or native dataclass validation (current)
- YAML anchor patterns: https://yaml.org/spec/1.2-old/spec.html#id2765878
- See `docs/MANUSCRIPT_WORKFLOW_REFERENCE.md` for parallelization details
- See `docs/PARALLEL_RUN_VALIDATION_CHECKLIST.md` for validation requirements
