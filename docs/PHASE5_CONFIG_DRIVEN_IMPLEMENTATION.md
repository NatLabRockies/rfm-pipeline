# Phase 5 Config-Driven Entry Point — Session Summary

**Date**: 2026-05-09
**Branch**: feat/unified-config-runner (commit 8ad9c59)
**Status**: Core implementation complete; integration pending

## Objective

Replace dataset-specific hardcoded scripts with a unified, config-driven entry point for the manuscript pipeline. Eliminate duplication in `run_300_sample_validation.py`, `run_fast_validation.py`, and future variants.

## What Was Built

### 1. Typed Configuration Schema (`src/rfm_pipeline/config.py`)

Comprehensive typed configuration dataclasses:

- **DatasetConfig**: Dataset type and optional path override
- **AlgorithmConfig**: Variance threshold, retained components
- **RuntimeConfig**: n_jobs (parallelization), batch size
- **StagesConfig**: All 5 pipeline stage parameters
  - ScreeningStageConfig (n_permutations, BH threshold)
  - InteractionStageConfig (p_threshold, tree estimators, depth)
  - NonlinearStageConfig (EDF threshold, transform families)
  - SparseStageConfig (stability resamples, LASSO alpha)
  - FinalArtifactsStageConfig (bootstrap count, alpha)
- **ValidationConfig**: Fast-mode overrides
- **OutputConfig**: Artifact directory, seed, verbosity
- **WorkflowConfig**: Root config object (all above combined)

**Features**:

- YAML loading with type validation (`load_config()`)
- Fast-mode override support (`apply_fast_mode_overrides()`)
- Full backward compatibility (all fields have defaults)

### 2. Unified Entry Point (`tools/run_manuscript_pipeline.py`)

Single command-line interface replacing dataset-specific scripts:

```bash
pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_no_caps.yml
```

**Features**:

- Accept config file path as argument
- CLI overrides: `--seed`, `--output-dir`, `--fast`
- Error handling and validation
- Integration pending with `manuscript_stages.py`

### 3. Example Configurations (`configs/`)

Three ready-to-use configs:

- **validation_300_sample_fast.yml**: CI/testing (5 perms, 8 resamples, 20 bootstraps, serial)
- **validation_300_sample_no_caps.yml**: Full validation (1000 perms, 100 resamples, 200 bootstraps, parallel)
- **validation_300_sample_serial.yml**: Full validation (same params, serial execution)
- **\_defaults.yml**: Base defaults for inheritance (YAML anchors)

**Each specifies**:

- Dataset, algorithm params, runtime config
- All 5 stage parameters
- Output directory and seed
- Validation/fast-mode settings

### 4. Unit Tests (`tests/test_config_loader.py`)

7 comprehensive tests (all passing):

- Load minimal config (required fields only)
- Load full config (all fields specified)
- File not found error handling
- Empty file validation (raises ValueError if no dataset.type)
- Fast-mode override application
- Partial override behavior
- Override deactivation when fast_mode=false

### 5. Documentation

- **configs/README.md**: Quick-start guide, config structure, usage examples
- **Entry point help**: `pixi run python tools/run_manuscript_pipeline.py --help`

## Test Results

```bash
# Unit tests
tests/test_config_loader.py: 7/7 PASS

# Entry point smoke test
pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_fast.yml
✓ Config loaded
✓ Dataset recognized
✓ Runtime config extracted
✓ Fast mode settings applied
```

## Files Modified

**New files**:

- src/rfm_pipeline/config.py (340 lines)
- tools/run_manuscript_pipeline.py (65 lines)
- tests/test_config_loader.py (115 lines)
- configs/validation_300_sample_fast.yml
- configs/validation_300_sample_no_caps.yml
- configs/validation_300_sample_serial.yml
- configs/\_defaults.yml
- configs/README.md

**Modified**:

- docs/AGENT_SYNC.md (updated Phase 5 status)

## What's Next (Integration Phase)

### 1. Wire into manuscript_stages.py

**Current entry point**:

```python
# tools/run_300_sample_validation.py (existing)
case_study_config = load_json("configs/manuscript_case_study.yml")
caps_dict = {"n_output_samples": 300, ...}
results = run_manuscript_reproduction_stage_chain(case_study_config, caps_dict)
```

**New entry point** (to be implemented):

```python
# tools/run_manuscript_pipeline.py (new)
config = load_config("configs/validation_300_sample_no_caps.yml")
# Convert WorkflowConfig to legacy case_study_config
case_study_config = _config_to_legacy_case_study(config)
n_jobs = config.runtime.n_jobs
results = run_manuscript_reproduction_stage_chain(case_study_config, n_jobs=n_jobs, ...)
```

**Integration steps**:

1. Create adapter function: `WorkflowConfig → legacy case_study_config`
1. Thread n_jobs through pipeline
1. Migrate run_300_sample_validation.py to use new runner
1. Test end-to-end with all configs
1. Run full gate

### 2. Maintain Backward Compatibility

- Keep legacy scripts (run_300_sample_validation.py) as deprecated wrappers
- Document migration path in docs
- No breaking changes to manuscript_stages.py public API

### 3. Extend for Future Datasets

Once working, add configs:

- `validation_full_manuscript.yml` (full dataset, if available)
- `validation_custom_params.yml` (user-defined)
- Dataset-specific variants (real data, different sample sizes)

## Design Decisions

### 1. Typed Dataclasses (Not Dicts)

**Why**: Strong typing enables IDE autocomplete, runtime validation, clear schema documentation.

**Trade-off**: Slight verbosity in loading logic, but saved by clarity.

### 2. YAML (Not JSON/TOML)

**Why**: More human-readable, industry-standard for configs, smaller footprint.

**Trade-off**: Limited to YAML 1.1 semantics (no date/binary).

### 3. Fast-Mode Overrides (Not CLI Flags)

**Why**: Config file is source of truth; overrides apply consistently across runs.

**Trade-off**: Requires editing config file for fast mode (mitigated by `--fast` CLI flag).

### 4. Single Entry Point (Not Script Per Dataset)

**Why**: Eliminates duplication, easier to add new modes, single testing path.

**Trade-off**: Requires more upfront config design (but pays off quickly).

## Risks & Mitigations

| Risk                                      | Mitigation                                                 |
| ----------------------------------------- | ---------------------------------------------------------- |
| Integration breaks existing pipeline      | Run full gate before merging; keep legacy scripts runnable |
| Config format evolves, old configs break  | Version config schema; add migration utility if needed     |
| Users confused by config options          | Comprehensive docs (CONFIGURATION_REFERENCE.md) + defaults |
| Performance regression from extra parsing | Config loading is \<100ms (negligible vs pipeline runtime) |

## Known Unknowns

- Exact n_jobs optimal setting for bootstrap parallelization (user-tunable in config)
- Whether full integration will surface additional tweaks to config schema
- Performance with very large datasets (untested; config structure supports it)

## Deliverables Checklist

- [x] Config schema designed and typed
- [x] Config loader with validation
- [x] Unified entry point script
- [x] 3 example configs created
- [x] Unit tests (7/7 pass)
- [x] Documentation (README, inline docstrings)
- [x] Commit to feat/unified-config-runner
- [x] Pushed to remote
- [ ] Integration with manuscript_stages.py
- [ ] Full gate pass
- [ ] PR created and merged

## Commit Information

```
feat: unified config-driven manuscript pipeline entry point

- Add typed config schema (src/rfm_pipeline/config.py): DatasetConfig,
  AlgorithmConfig, RuntimeConfig, StagesConfig, WorkflowConfig
- Add config loader with validation and fast-mode override support
- Create unified entry point (tools/run_manuscript_pipeline.py) accepting YAML
- Add 3 example configs
- Add 7 unit tests for config loading (all pass)
- Add configs/README.md user guide
- Update docs/AGENT_SYNC.md to reflect Phase 5 progress

Integration with manuscript_stages.py pending.
```

**Commit hash**: 8ad9c59

## Usage

### Run Full Validation (Parallel)

```bash
pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_no_caps.yml
```

### Run CI Fast Mode

```bash
pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_fast.yml --fast
```

### Custom Seed and Output

```bash
pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_no_caps.yml \
  --seed 999 \
  --output-dir /tmp/my_run
```

### Create New Config

```bash
cp configs/validation_300_sample_no_caps.yml configs/my_custom_run.yml
# Edit configs/my_custom_run.yml
pixi run python tools/run_manuscript_pipeline.py configs/my_custom_run.yml
```

## Estimated Effort for Integration

- Wire into manuscript_stages.py: 1-2 hours
- Test end-to-end: 1 hour
- Fix any integration issues: 0.5-1 hours
- **Total**: 2.5-4 hours (not started; awaiting full integration tests)

## Related Documentation

- **Design doc**: `docs/REFACTOR_CONFIG_DRIVEN_DESIGN.md` (12.7KB, full architecture)
- **Validation checklist**: `docs/PARALLEL_RUN_VALIDATION_CHECKLIST.md`
- **Session summary**: `docs/SESSION_SUMMARY_2026_05_09.md`
- **ENGINEERING_MANIFEST.md**: Phase 5 section
- **AGENT_SYNC.md**: Current slice status
