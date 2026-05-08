#!/usr/bin/env python3
"""Add scenario factors (AFSC, UAEORO, scenario, run_id) to datasets."""

from pathlib import Path

import numpy as np
import pandas as pd


def add_scenario_factors():
    """Create datasets with all scenario factors from old full data."""
    print("Loading old full data with scenario factors...")
    X_old = pd.read_parquet("docs/final_scripts_from_hpc/sa_068.null_010.X.nl.parquet")
    Y_old = pd.read_parquet("docs/final_scripts_from_hpc/sa_068.null_010.Y.parquet")
    print(f"Old data: X={X_old.shape}, Y={Y_old.shape}")

    # Get base input features
    input_meta = pd.read_parquet(
        "docs/final_scripts_from_hpc/model_artifacts/final_ols_model/"
        "postfit_diagnostics/all_input_metadata.parquet"
    )
    base_features = input_meta["input_name"].tolist()
    print(f"Base features: {len(base_features)}")

    # Create sample_id
    X_old["sample_id"] = X_old["scenario"].astype(str) + "_" + X_old["run_id"].astype(str)
    Y_old["sample_id"] = Y_old["scenario"].astype(str) + "_" + Y_old["run_id"].astype(str)

    # Build X: sample_id + scenario + run_id + base features
    # Note: AFSC and UAEORO are already in base_features list from input_metadata
    X_cols = ["sample_id", "scenario", "run_id"] + base_features
    X_new = X_old[X_cols].copy()
    print(f"X columns (first 10): {X_new.columns.tolist()[:10]}")

    # Build Y: sample_id + all outputs
    Y_cols = [c for c in Y_old.columns if c not in ["scenario", "run_id"]]
    Y_new = Y_old[Y_cols].copy()

    print(f"New: X={X_new.shape}, Y={Y_new.shape}")

    # Save full 20k dataset
    output_dir = Path("artifacts/preprocessed_real_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    X_new.to_parquet(output_dir / "sample_7500.X.parquet", index=False)
    Y_new.to_parquet(output_dir / "sample_7500.Y.parquet", index=False)
    print(f"✓ Saved {len(X_new)} samples to {output_dir}/")

    # Create 3k test subset
    print("\nCreating 3k test subset...")
    np.random.seed(42)
    sample_ids = []
    for scen in X_new["scenario"].unique():
        ids = X_new[X_new["scenario"] == scen]["sample_id"].values
        sampled = np.random.choice(ids, size=750, replace=False)
        sample_ids.extend(sampled)

    X_test = X_new[X_new["sample_id"].isin(sample_ids)].copy()
    Y_test = Y_new[Y_new["sample_id"].isin(sample_ids)].copy()

    # Create holdout assignments
    holdout = pd.DataFrame({"sample_id": X_test["sample_id"].values})
    np.random.seed(42)
    holdout["split"] = np.random.choice(["train", "test"], size=len(holdout), p=[0.9, 0.1])

    test_dir = Path("artifacts/test_dataset_3k")
    test_dir.mkdir(parents=True, exist_ok=True)
    X_test.to_parquet(test_dir / "X.parquet", index=False)
    Y_test.to_parquet(test_dir / "Y.parquet", index=False)
    holdout.to_parquet(test_dir / "holdout_assignments.parquet", index=False)

    print(f"✓ Saved 3k test to {test_dir}/")
    train_count = int((holdout["split"] == "train").sum())
    test_count = int((holdout["split"] == "test").sum())
    print(f"  Train: {train_count}, Test: {test_count}")


if __name__ == "__main__":
    add_scenario_factors()
