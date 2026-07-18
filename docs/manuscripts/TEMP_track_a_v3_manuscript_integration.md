# TEMP — Track A v3 predictive add-on integration notes

Date: 2026-06-29
Scope: Final single-track predictive add-on framing for manuscript (Track A v3 only)
Status: Temporary handoff note for manuscript editing

## 1) What was actually implemented

Implemented model/run (final):

- `scripts/fit_track_a_v3.py`
- `src/rfm_pipeline/track_a_v3.py`
- `scripts/submit_track_a_v3.sh`
- Slurm job: `14801489` (Kestrel, shared, 24h, 220G, 104 CPUs) — **COMPLETED**

Track A v3 capabilities implemented:

1. kNN imputation of BSM measurement features from nearest synthetic rows in knob-space.
1. Near-BSM sample weighting for hybrid/RF training.
1. OOD diagnostics + OOD-scaled 95% uncertainty interval for BSM nRMSE estimate.
1. Runtime residual modeling with added large-output regime term.

## 2) What was actually demonstrated (final run outputs)

Primary output file:

- `artifacts/sensitivity/wave5_measurement_models_v3/wave5_track_a_v3_summary.json`

Demonstrated metrics from final run:

- Group-CV quality fit (nRMSE, hybrid): `R^2 = 0.7542`, `RMSE = 0.00854`
- BSM estimate (final model):
  - Predicted nRMSE: `0.08788`
  - Observed nRMSE: `0.07210`
  - Percent error: `+21.89%`
  - 95% interval: `[0.06435, 0.11141]`
  - OOD nearest-distance: `2.700`
- Runtime model (same final run): `R^2 = -1.155` (not suitable for point-accuracy claims)

Planning-band analytics generated from final outputs:

- `artifacts/sensitivity/wave5_measurement_models_v3/track_a_v3_planning_summary.json`
- `artifacts/sensitivity/wave5_measurement_models_v3/track_a_v3_decision_bands.csv`

## 3) New figures/data prepared for manuscript update

Prepared and copied into `docs/manuscripts/`:

- `fig_track_a_v3_bsm_planning_interval.svg` (+ `.pdf`)
  - Shows final BSM point estimate, 95% interval, observed BSM marker, and planning bands.
- `fig_track_a_v3_cv_scatter.svg` (+ `.pdf`)
  - Shows in-domain observed-vs-predicted CV fit for the final hybrid model.

Supporting generation script:

- `scripts/build_track_a_v3_manuscript_addon.py`

## 4) How to incorporate into manuscript (short, non-overclaiming)

### 4.1 Positioning statement (single-track, short)

Use Track A v3 as an uncertainty-aware **workflow planning add-on**:

- Intended use: triage/ranking/planning
- Not intended use: precise absolute BSM prediction

### 4.2 Results paragraph guidance (final model only)

Include only final model evidence (no v1/v2 narrative):

- Report hybrid CV fit (`R^2 = 0.7542`) as in-domain support.
- Report BSM point+interval (`0.0879 [0.0644, 0.1114]`) vs observed (`0.0721`).
- State practical interpretation as planning guidance under uncertainty.

### 4.3 Runtime wording constraint

Because runtime `R^2 < 0`, runtime text should be restricted to coarse regime guidance only:

- fast / moderate / slow style planning categories
- no strong point-accuracy wording

### 4.4 Suggested compact manuscript wording (drop-in)

> We include a single uncertainty-aware transfer model (Track A v3) as a workflow-planning add-on. The model is used to rank and triage candidate configurations rather than to provide precise absolute BSM forecasts. On grouped cross-validation it shows strong in-domain fit for nRMSE (`R^2=0.754`), and for the BSM case yields `0.0879` with a 95% interval of `[0.0644, 0.1114]` (observed `0.0721`), supporting use as planning guidance under uncertainty.

### 4.5 Caption guidance

For `fig_track_a_v3_bsm_planning_interval`:

> Track A v3 planning estimate for BSM with 95% uncertainty interval and decision bands. This add-on is intended for workflow triage/planning, not precision prediction.

For `fig_track_a_v3_cv_scatter`:

> Grouped cross-validation observed-vs-predicted nRMSE for Track A v3 (hybrid model), shown to document in-domain fit quality for the planning add-on.

## 5) Evidence file list to cite internally

- `artifacts/sensitivity/wave5_measurement_models_v3/wave5_track_a_v3_summary.json`
- `artifacts/sensitivity/wave5_measurement_models_v3/wave5_track_a_v3_predictions.csv`
- `artifacts/sensitivity/wave5_measurement_models_v3/track_a_v3_planning_summary.json`
- `docs/manuscripts/fig_track_a_v3_bsm_planning_interval.pdf`
- `docs/manuscripts/fig_track_a_v3_cv_scatter.pdf`
