# 300-Sample Parallel Run Validation Checklist

**Command executed**: `pixi run python tools/run_300_sample_validation.py --no-caps`
**Started**: 2026-05-09 07:47:10
**Expected duration**: 2-4 hours (vs ~390 seconds serial on baseline machine)
**Artifacts location**: `artifacts/validation_300_sample_no_caps/`

______________________________________________________________________

## Acceptance Criteria: Artifact Counts

| Artifact                            | 300-Sample Expected | Manuscript Reference | Status                                          |
| ----------------------------------- | ------------------- | -------------------- | ----------------------------------------------- |
| **Output Conditioning**             |                     |                      |                                                 |
| PCA components retained             | 27                  | 39                   | Expected (smaller training set)                 |
| Output terms retained               | 9,466 / 23,495      | —                    | Verify count > 0                                |
| **Empirical Null Screening**        |                     |                      |                                                 |
| Terms retained (after null filter)  | ~300-350            | 349                  | Expected ±10%                                   |
| Variance threshold                  | 90%                 | 90%                  | Verify exact match                              |
| **Interaction Discovery**           |                     |                      |                                                 |
| Interaction pairs discovered        | ~800-900            | 367                  | Expected (more pairs from 27 → more candidates) |
| Pair retention rate                 | ~45%                | 12.7% (248/1953)     | Verify calculation                              |
| **Nonlinear Discovery**             |                     |                      |                                                 |
| Transformations discovered          | ~150-170            | 112                  | Expected (same cascade effect)                  |
| Transformation retention rate       | ~95%                | —                    | Verify >90%                                     |
| **Sparse Selection (if completes)** |                     |                      |                                                 |
| Candidate features                  | ~1,300              | —                    | Verify count in range                           |
| Resampling iterations completed     | 100                 | 100                  | Verify all 100 resamples run                    |
| **Final Artifacts (if completes)**  |                     |                      |                                                 |
| Final feature count                 | ~1,300+             | 352                  | Only if sparse selection completes              |

______________________________________________________________________

## Stage Completion Status

Check `stage_summaries.json` in artifacts directory:

```json
{
  "output_conditioning": {"status": "complete", "n_retained_outputs": 9466, "n_components": 27},
  "empirical_null_screening": {"status": "complete", "n_retained_terms": 301},
  "interaction_discovery": {"status": "complete", "n_pairs_retained": 884},
  "nonlinear_discovery": {"status": "complete", "n_transformations": 160},
  "sparse_selection": {"status": "?" , "n_retained_features": "?"},
  "final_artifacts": {"status": "?"}
}
```

**Expected completion order**:

1. ✅ Output conditioning (5s)
1. ✅ Empirical null screening (8s with parallelization)
1. ✅ Interaction discovery (15s with parallelization)
1. ✅ Nonlinear discovery (40s with parallelization)
1. ? Sparse selection (likely bottleneck; 60-180s depending on LASSO efficiency)
1. ? Final OLS (10-20s if sparse completes)

______________________________________________________________________

## Performance Metrics to Capture

From stdout or log file:

- **Wall time per stage** (compare serial vs parallel)
- **Speedup factor**: (serial time / parallel time)
- **Peak memory usage**: Monitor during sparse resampling
- **CPU utilization**: % of cores active during each parallelizable stage
- **Number of workers active**: Should match `n_jobs=-1` (all CPUs)

**Expected speedups**:

- Empirical null: ~3-4× (B+1=21 permutations on 4-core)
- Interaction discovery: ~3-4× (B+1=21 permutations)
- Nonlinear discovery: ~4-8× (250+ base_feature × family pairs on 4-core)
- Sparse selection: ~2-3× (100 resamples; LASSO itself not parallelizable)

______________________________________________________________________

## QA Audit Status

Check `qa_audit_summary.json`:

```text
{
  "qa_status": "pass|fail",
  "n_artifacts": "<count>",
  "n_missing_artifacts": 0,
  "n_empty_artifacts": 0,
  "n_failed_metric_checks": 0,
  "stage_summaries": {...}
}
```

**Must have**:

- `qa_status == "pass"`
- `n_missing_artifacts == 0`
- `n_empty_artifacts == 0`
- `n_failed_metric_checks == 0`

______________________________________________________________________

## File Integrity Checks

After run completes:

```bash
# Count stage summary files
ls -la artifacts/validation_300_sample_no_caps/ | grep "\.json" | wc -l
# Expected: stage_summaries.json, qa_audit_summary.json, metrics.json + data files

# Verify all stage directories exist
ls -1 artifacts/validation_300_sample_no_caps/ | grep -E "^(output_conditioning|empirical_null|interaction|nonlinear|sparse|final)" | wc -l
# Expected: at least 4-5 directories (sparse/final may not exist if stages incomplete)

# Check for error logs
find artifacts/validation_300_sample_no_caps/ -name "*.error" -o -name "*.log"
# Expected: None (or only info-level logs)
```

______________________________________________________________________

## Comparison to Prior No-Caps Run

Prior session's no-caps run (PID 16878) achieved:

| Stage                 | Prior Session Result          | This Run | Match? |
| --------------------- | ----------------------------- | -------- | ------ |
| output_conditioning   | 27 components, 9,466 outputs  | ?        |        |
| empirical_null_screen | 301 terms                     | ?        |        |
| interaction_discovery | 884 pairs                     | ?        |        |
| nonlinear_discovery   | 160 transformations           | ?        |        |
| sparse_selection      | INCOMPLETE (killed after 11h) | ?        |        |
| final_artifacts       | INCOMPLETE                    | ?        |        |

**Success criteria**: All counts match OR can be explained by minor code differences (e.g., randomization seed, iteration count adjustments).

______________________________________________________________________

## Validation Commands (Post-Completion)

Once run finishes:

```bash
# 1. Examine stage summaries
cat artifacts/validation_300_sample_no_caps/stage_summaries.json | python -m json.tool

# 2. Check QA audit
cat artifacts/validation_300_sample_no_caps/qa_audit_summary.json | python -m json.tool

# 3. Verify reproducibility with seed=0
pixi run python tools/run_300_sample_validation.py --no-caps --seed 0

# 4. Run full gate to ensure no regressions
bash ./test_repo.sh --check
```

______________________________________________________________________

## Next Action (Upon Completion)

Once parallel run finishes:

1. **Capture start/end times and wall time per stage**
1. **Compare artifact counts to table above**
1. **Check QA status == "pass"**
1. **If all pass**: Green light for config-driven refactor milestone
1. **If any fail**: Debug and rerun before proceeding to refactor

______________________________________________________________________

## Reference: Manuscript Algorithm Parameters

From `configs/manuscript_case_study.yml`:

```yaml
case_study:
  retained_components: 39          # PCA target
  variance_threshold: 0.90          # PCA cutoff
  n_permutations: 1000             # B+1 empirical null permutations
  bh_q_threshold: 0.10             # Benjamini-Hochberg FDR threshold
  edf_threshold: 2.5               # Nonlinear EDF cutoff
  p_threshold: 0.05                # Interaction p-value threshold
  n_stability_subsamples: 100      # Sparse stability resamples
```

Note: 300-sample dataset will reach 90% variance threshold at fewer components (27 vs 39) due to smaller training set, cascading to higher interaction/nonlinear discovery counts. This is expected and correct.

______________________________________________________________________

## Dependencies

For artifact validation to proceed:

- [ ] Parallel run must complete without OOM, process kill, or timeout
- [ ] All 4 main stages must reach "complete" status
- [ ] QA audit must show "pass"
- [ ] Artifact counts must be reasonable (match or exceed expectations)

If these conditions met → proceed to refactor milestone (config-driven entry point)

If these conditions not met → debug, fix, rerun before refactor
