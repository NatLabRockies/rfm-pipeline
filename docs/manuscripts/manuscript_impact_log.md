# Manuscript Impact Log

Running ledger of code/config/data changes that affect the JDS BSM
manuscript (`jds_bsm_v22.tex`) or that require re-running upstream
artifacts. Update this file every time a change lands that:

1. invalidates a published or staged result,
1. forces a re-fit / re-run / re-export,
1. changes a value cited in the manuscript text or tables, **or**
1. adds a new figure, table, or numeric claim that should be reflected
   in the manuscript.

Entries are newest-first. Each entry records: date, scope, what
changed, manuscript impact, and the re-run / re-export action
required.

______________________________________________________________________

## 2026-06-06 — Round-4 adversarial audit

### 🟢 No manuscript-value impact

Round-4 fixes were all developer-experience / drift-prevention. No
shipped manuscript number changes; no re-run required.

- **rfm-pipeline (caeed3a).** `transform_families` now **raises**
  instead of silently dropping (root-cause fix for round-3 finding).
  `configs/manuscript_case_study_fast_sparse.yml` renamed to
  `configs/manuscript_case_study.yml` so the canonical name every
  docstring already referenced actually exists on disk. Docs build
  (`pixi run docs`) restored. `paper.md` now discloses the synthetic-DGP
  sensitivity baseline holds `n_tree_estimators` / `max_tree_depth`
  below manuscript values for LHS-budget reasons (BSM case study
  retains full manuscript baselines — no result impact).
- **bsm-public-rf (pending commit).** 8 HPC configs were carrying the
  dead `lasso_alpha_percentile` key (would crash on
  `load_config` against pinned rfm-pipeline a6c18e6) and 8 also
  carried the dead `transform_families` key (would crash against
  caeed3a). All cleaned. `pixi.lock` refreshed; pin bumped to
  caeed3a. New test `tests/test_configs_loadable.py` (parametrized
  over 16 standalone HPC YAMLs) prevents this drift recurring.
- **Prediction-equation docs corrected.**
  `artifacts/final_model/README.md` and `README_intercept.md` had the
  raw-scale prediction rule wrong (double-divided by `scale_x`, and
  falsely claimed `scale_y == 1` for all outputs). Corrected. No
  shipped CSV changed — the formula in the docs disagreed with the
  numbers; numbers were right. **Reviewers who computed predictions
  from the broken formula would have gotten values inflated by
  `scale_x` per term; flag in any released errata.**

**Re-run action:** none.
**Manuscript text edits:** none.

______________________________________________________________________

### 🔴 B1. `lasso_alpha_percentile` was a dead config knob

- **What changed.** The `SparseStageConfig.lasso_alpha_percentile`
  field was declared and swept by the sensitivity study over
  `(10, 20, 40, 60, 80)`, but the EBIC LASSO selector
  (`_select_component_lasso_by_ebic`) hardcoded the grid as
  `np.geomspace(..., num=40)` and never read the config field.
  Renamed to `lasso_alpha_grid_size` and threaded into the selector
  (rfm-pipeline `a6c18e6`).
- **Manuscript impact.**
  - Section 6 / Table 4 / Table 5 / Figure 7: any RF importance or
    polynomial-meta-regression coefficient attributed to
    "LASSO α percentile" in the **existing** sensitivity wave1/2/3
    results is variance on a constant feature. It must be **dropped**
    from the meta-regression design matrix when fitting against the
    legacy CSVs.
  - Table 4 currently lists this row as
    `LASSO EBIC grid size  40  [TBD]`. After the rename, the swept
    values are `(20, 40, 80, 160)` and the manuscript [TBD] cell can
    be filled in once the **next** sensitivity wave runs with the
    fixed code.
- **Re-run required.**
  - **Meta-regression refit.** Re-run `scripts/fit_meta_regression.py`
    and `scripts/plot_sensitivity_results.py` against the existing
    `wave{1,2,3}_results.csv` **after dropping the
    `stages.sparse_selection.lasso_alpha_percentile` column** (now
    `lasso_alpha_grid_size`, but for legacy data the column is a
    constant noise dimension). Re-export `artifacts/sensitivity/ wave1_rf_quality.pkl` and `wave1_rf_runtime.pkl`.
  - **Future sensitivity wave (optional).** To populate the Table 4
    `[TBD]` cell, run a fresh sensitivity wave that actually varies
    the grid size. This is optional — the manuscript currently lists
    `[TBD]` and can remain so until the user is ready.
- **Manuscript text edits required.**
  - Table 5 / Figure 7 caption: re-state the predictor list excluding
    `lasso_alpha_percentile` (or replace with the corrected
    `lasso_alpha_grid_size` once new data exist).
  - Section 6.1 text quoting per-predictor importances must be
    re-derived from the refit.

### 🟢 H7. Sensitivity sweep `delta_threshold_override` `None` removed

- **What changed.** The sensitivity LHS sweep over
  `delta_threshold_override` was `(0.001, 0.002, 0.005, 0.010, None)`;
  the manuscript Table 4 row lists exactly the first four values.
  `None` disabled pruning entirely (a configuration the manuscript
  never reports) and was hit on ≈20% of the sensitivity runs.
  Removed `None` (rfm-pipeline `a6c18e6`).
- **Manuscript impact.**
  - Existing wave1/2/3 CSVs contain ≈20% rows where
    `delta_threshold_override` is null (disable-pruning runs). The
    meta-regression should either (a) drop those rows or
    (b) keep them and document them in Table 4 as a fifth
    "pruning disabled" sweep value.
- **Re-run required.**
  - **Meta-regression refit:** decide between (a) and (b) and re-fit
    accordingly. Recommended (a): drop disable-pruning rows for
    consistency with manuscript Table 4.

### 🟡 H1. Transform library reduced to 4 manuscript families

- **What changed.** `EXPONENTIAL` removed from public `__all__` and
  from `KNOWN_TRANSFORMATIONS`; manuscript §3.5 lists 4 families
  only (quadratic, logarithmic, inverse, square-root). The
  `DEFAULT_TRANSFORM_LIBRARY` was already 4 families since round 2.
- **Manuscript impact.** None — manuscript already specifies 4
  families. Round-2 already corrected the published default. This
  change just cleans up the public API to match.
- **Re-run required.** None.

### 🟡 B3. Jaccard / Spearman thresholds now in typed config

- **What changed.** `jaccard_threshold = 0.75` and
  `spearman_threshold = 0.90` are now first-class fields on
  `SparseStageConfig` (rfm-pipeline `a6c18e6`).
- **Manuscript impact.** None — values match manuscript Table 1
  baselines. Previously only reachable via undocumented
  `case_study` dict path; now reachable via documented YAML schema.
- **Re-run required.** None.

### 🟢 H4. `manuscript_alignment_audit.md` BH q typo fixed

- **What changed.** Audit doc said BH `q = 0.10`; code default and
  manuscript Table 1 specify `q = 0.05`. Doc fixed.
- **Manuscript impact.** None — actual pipeline already used 0.05.
- **Re-run required.** None.

______________________________________________________________________

## 2026-06-05 — Round-2 adversarial audit (kept for traceability)

### 🟡 Round-2 transform library reduction

- **What changed.** `DEFAULT_TRANSFORM_LIBRARY` reduced from 5 to 4
  families (dropped `exp(x)`); this was the round-2 fix that aligned
  the implementation with manuscript §3.5.
- **Manuscript impact.** None — manuscript already specified 4
  families. This change made the implementation match.
- **Re-run required.** None for the published case-study run (which
  used 4 families regardless). Any **future** full-dataset re-run
  will use the same 4 families, so artifact counts continue to match
  the manuscript (172 enriched / 132 final / 41 transforms / 29 in
  final support).

### 🟡 Round-2 `final_ols_summary.csv` reference columns

- **What changed.** The `manuscript_final_predictor_count_reference`
  and `manuscript_final_ols_holdout_nrmse_reference` columns now
  carry the v22 values (132, 0.0721).
- **Manuscript impact.** None — same values as manuscript Table 2.
  This change protects against regression: tripwire assertions in the
  pipeline now compare against the same constants the manuscript
  cites.
- **Re-run required.** None.

______________________________________________________________________

## Open / pending manuscript actions

The following items affect the manuscript but are **waiting on the
user** (not Copilot-completable):

- **paper.md (rfm-pipeline JOSS submission):** real ORCID, real
  submission date, real DOE BETO award identifier, NREL vs NLR
  affiliation consistency between `paper.md`, `pyproject.toml`,
  `CITATION.cff`, and the manuscript front matter.
- **Manuscript Table 4 `[TBD]` cells:**
  - `Interaction null-quantile threshold` — baseline 0.995, swept
    [TBD]. Code currently only uses the baseline; no sweep planned.
  - `LASSO EBIC grid size` — baseline 40, swept [TBD]. Now properly
    threaded; a future sensitivity wave with `(20, 40, 80, 160)` can
    fill the cell.
- **Sensitivity meta-regression refit:** the existing wave1/2/3 CSVs
  have one no-op column (`lasso_alpha_percentile`) and ≈20% disabled-
  pruning rows. A refit is required before any updated Section 6
  tables, figures, or numeric claims can be re-cited in the
  manuscript. See B1 / H7 above.

______________________________________________________________________

## How to use this log

- Add a new entry under a dated heading every time a change lands
  that meets one of the four criteria at the top.
- Use the 🔴 / 🟡 / 🟢 markers to flag severity:
  - 🔴 invalidates a published result; refit / re-export required
  - 🟡 changes the implementation to match the manuscript or adds
    new infrastructure; no re-fit required
  - 🟢 doc / typo / hygiene fix with no scientific impact
- When a referenced refit completes, edit the entry to add a final
  "**Resolved:** <date> — <commit>" line rather than deleting the
  entry.
