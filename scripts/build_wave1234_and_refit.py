#!/usr/bin/env python3
# Copyright (c) 2026 Dylan Hettinger
"""Build wave1234 from wave123 + wave4, refit RF, regenerate figures.

Wave4 was collected with a newer (slimmer) collect_sensitivity_results.py
that drops 3 columns (stdout_tail, n_features_retained, null_screened) and
uses the spec-level family label (calibrated_structure) instead of the
downstream rename (bsm_structure). This script:

  1. Loads wave123_combined_clean.csv (6,225 rows, 61 cols, family = bsm_structure)
     and wave4_results.csv (33 rows, 58 cols, family = calibrated_structure).
  2. Pads wave4 with NaN for the 3 missing columns; renames calibrated_structure
     → bsm_structure on the calibrated block (3 pure_synthetic rows kept as-is).
  3. Filters wave4 by the wave123 cleaning recipe (delta_threshold_override
     notna() & final_ols_nrmse notna()).
  4. Concatenates → wave1234_combined_clean.csv.
  5. Calls fit_sensitivity_rf.py to refit the RFs (writes wave1234 pkls).
  6. Calls compute_bsm_rf_validation.py against the wave1234 inputs.

Re-run via:
    pixi run python scripts/build_wave1234_and_refit.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/sensitivity"


def main() -> int:
    """Merge wave123 + wave4, refit, and emit wave1234 artifacts."""
    w123 = pd.read_csv(ART / "wave123_combined_clean.csv")
    w4 = pd.read_csv(ART / "wave4_results.csv")
    print(f"# wave123: {w123.shape}; wave4: {w4.shape}")

    w4 = w4.copy()
    w4["dgp_family"] = w4["dgp_family"].replace({"calibrated_structure": "bsm_structure"})
    for c in ("stdout_tail", "n_features_retained", "null_screened"):
        if c not in w4.columns:
            w4[c] = pd.NA
    w4 = w4[list(w123.columns)]

    dth = "stages.final_artifacts.delta_threshold_override"
    keep = w4[dth].notna() & w4["final_ols_nrmse"].notna()
    w4_clean = w4.loc[keep].copy()
    print(f"# wave4 after filter: {len(w4_clean)} / {len(w4)}")

    w1234 = pd.concat([w123, w4_clean], ignore_index=True)
    out_csv = ART / "wave1234_combined_clean.csv"
    w1234.to_csv(out_csv, index=False)
    print(f"# wrote {out_csv}: {w1234.shape}")
    print(f"  family counts: {w1234['dgp_family'].value_counts().to_dict()}")
    b = w1234[w1234["dgp_family"] == "bsm_structure"]
    print(f"  bsm rows: {len(b)} (was {(w123['dgp_family'] == 'bsm_structure').sum()})")

    # Refit RFs (writes wave1234_rf_quality.pkl / wave1234_rf_runtime.pkl
    # into ART because fit_sensitivity_rf.py keys the prefix on the leading
    # underscore-token of the input stem).
    print("\n# refitting RFs on wave1234 ...")
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts/fit_sensitivity_rf.py"),
            "--results",
            str(out_csv),
            "--top-n",
            "9",
            "--dump-models",
            str(ART),
        ]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
