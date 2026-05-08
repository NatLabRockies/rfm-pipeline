#!/usr/bin/env python3
r"""Generate feature catalog with first-order, interaction, and nonlinear candidates.

This utility creates the feature catalog required by the manuscript reproduction workflow.
The catalog specifies which features, interactions, and nonlinear transforms the workflow
should evaluate during screening and discovery stages.

Usage:
    # Generate catalog with all pairwise interactions (WARNING: very large for >100 features)
    python scripts/generate_feature_catalog.py \\
        --input-matrix artifacts/my_data/X.parquet \\
        --output-catalog artifacts/my_data/feature_catalog.parquet \\
        --interaction-strategy all

    # Generate catalog with top SHAP-scored interaction candidates (recommended)
    python scripts/generate_feature_catalog.py \\
        --input-matrix artifacts/my_data/X.parquet \\
        --output-matrix artifacts/my_data/Y.parquet \\
        --output-catalog artifacts/my_data/feature_catalog.parquet \\
        --interaction-strategy top-shap \\
        --max-interactions 500

    # Generate catalog with domain-specified interactions from file
    python scripts/generate_feature_catalog.py \\
        --input-matrix artifacts/my_data/X.parquet \\
        --output-catalog artifacts/my_data/feature_catalog.parquet \\
        --interaction-strategy from-file \\
        --interaction-file artifacts/my_data/interaction_pairs.csv

    # First-order features only (interactions discovered dynamically)
    python scripts/generate_feature_catalog.py \\
        --input-matrix artifacts/my_data/X.parquet \\
        --output-catalog artifacts/my_data/feature_catalog.parquet \\
        --interaction-strategy none

Notes
-----
    - Input matrix must have sample_id as first column
    - Special columns (sample_id, scenario, run_id, AFSC, UAEORO) are excluded from catalog
    - Nonlinear transforms added for features with appropriate domains
    - For large feature sets (>200), use 'top-shap' or 'from-file' interaction strategies
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


def load_input_matrix(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load input matrix and extract feature names.

    Returns
    -------
        Tuple of (full dataframe, list of feature names excluding special columns)
    """
    X = pd.read_parquet(path)

    # Validate structure
    if "sample_id" not in X.columns:
        raise ValueError(
            f"Input matrix must have 'sample_id' as first column. "
            f"Found columns: {list(X.columns[:5])}"
        )

    # Exclude special columns
    special_cols = {"sample_id", "scenario", "run_id", "AFSC", "UAEORO"}
    feature_cols = [col for col in X.columns if col not in special_cols]

    print(f"Loaded input matrix: {X.shape}")
    print(f"  Total columns: {len(X.columns)}")
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Special columns: {len(special_cols & set(X.columns))}")

    return X, feature_cols


def generate_first_order_features(feature_names: list[str]) -> pd.DataFrame:
    """Generate first-order feature catalog entries.

    Filters out features that already have transform suffixes (_inverse, _log, _quadratic, _sqrt)
    as these are derived features, not base inputs.
    """
    transform_suffixes = ("_inverse", "_log", "_quadratic", "_sqrt")
    base_features = [
        f for f in feature_names if not any(f.endswith(suffix) for suffix in transform_suffixes)
    ]

    if len(base_features) < len(feature_names):
        n_filtered = len(feature_names) - len(base_features)
        print(
            "  Note: Filtered "
            f"{n_filtered} features with transform suffixes "
            f"(keeping {len(base_features)} base features)"
        )

    return pd.DataFrame(
        {
            "feature_name": base_features,
            "feature_type": ["numeric"] * len(base_features),
            "origin": ["model_factors"] * len(base_features),
        }
    )


def generate_all_interactions(feature_names: list[str]) -> pd.DataFrame:
    """Generate all pairwise interaction candidates."""
    pairs = list(combinations(feature_names, 2))

    if len(pairs) > 50000:
        print(
            f"\nWARNING: Generating {len(pairs):,} interaction pairs "
            f"from {len(feature_names)} features."
        )
        print("This will create a very large catalog and slow workflow runtime significantly.")
        print("Consider using --interaction-strategy top-shap with --max-interactions 500-1000.")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    return pd.DataFrame(
        {
            "feature_name": [f"{f1}:{f2}" for f1, f2 in pairs],
            "feature_type": ["interaction"] * len(pairs),
            "origin": ["model_factors"] * len(pairs),
        }
    )


def generate_top_shap_interactions(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    feature_names: list[str],
    max_pairs: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate interaction candidates using SHAP interaction values.

    Uses TreeSHAP on a gradient boosting model to rank all pairwise interactions,
    then selects the top N pairs by mean absolute SHAP interaction value.

    This provides a data-driven way to pre-select promising interaction candidates
    without evaluating all C(n,2) pairs.
    """
    print(f"\nRunning SHAP interaction analysis to select top {max_pairs} pairs...")
    print("This may take several minutes for large datasets.")

    try:
        import lightgbm as lgb
        import shap
    except ImportError as error:
        raise ImportError(
            "SHAP interaction analysis requires: lightgbm, shap\n"
            "Install with: pixi add lightgbm shap"
        ) from error

    # Use first 5000 samples if dataset is large
    n_samples = min(5000, len(X))
    if n_samples < len(X):
        print(f"Using {n_samples} samples for SHAP analysis (full dataset has {len(X)})")
        sample_indices = np.random.RandomState(random_seed).choice(len(X), n_samples, replace=False)
        X_sample = X.iloc[sample_indices]
        Y_sample = Y.iloc[sample_indices]
    else:
        X_sample = X
        Y_sample = Y

    # Prepare feature matrix
    special_cols = {"sample_id", "scenario", "run_id", "AFSC", "UAEORO"}
    X_features = X_sample[[col for col in X_sample.columns if col not in special_cols]]

    # Use first output component for SHAP ranking
    y_target = Y_sample.iloc[:, 0].values if len(Y_sample.shape) > 1 else Y_sample.values

    # Train gradient boosting model
    print("Training gradient boosting model...")
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=random_seed,
        verbose=-1,
    )
    model.fit(X_features, y_target)

    # Compute SHAP interaction values
    print("Computing SHAP interaction values...")
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_interaction_values = explainer.shap_interaction_values(X_features)

    # Rank all pairwise interactions by mean absolute SHAP interaction value
    n_features = len(feature_names)
    pair_scores = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            # SHAP interaction matrix is symmetric, use upper triangle
            interaction_values = shap_interaction_values[:, i, j]
            mean_abs_score = np.mean(np.abs(interaction_values))
            pair_scores.append(
                {
                    "feature_1": feature_names[i],
                    "feature_2": feature_names[j],
                    "mean_abs_shap": mean_abs_score,
                }
            )

    # Sort by score and take top N
    pair_scores_df = pd.DataFrame(pair_scores).sort_values("mean_abs_shap", ascending=False)
    top_pairs = pair_scores_df.head(max_pairs)

    print("\nTop 5 interaction pairs by SHAP score:")
    for _, row in top_pairs.head(5).iterrows():
        print(f"  {row['feature_1']} × {row['feature_2']}: {row['mean_abs_shap']:.6f}")

    # Create catalog entries
    return pd.DataFrame(
        {
            "feature_name": [
                f"{row['feature_1']}:{row['feature_2']}" for _, row in top_pairs.iterrows()
            ],
            "feature_type": ["interaction"] * len(top_pairs),
            "origin": ["model_factors"] * len(top_pairs),
        }
    )


def generate_interactions_from_file(interaction_file: Path) -> pd.DataFrame:
    """Load interaction pairs from CSV file.

    File must have columns: feature_1, feature_2
    Each row specifies one interaction pair to include in catalog.
    """
    pairs_df = pd.read_csv(interaction_file)

    required_cols = {"feature_1", "feature_2"}
    if not required_cols.issubset(pairs_df.columns):
        raise ValueError(
            f"Interaction file must have columns: {required_cols}. Found: {set(pairs_df.columns)}"
        )

    return pd.DataFrame(
        {
            "feature_name": [
                f"{row['feature_1']}:{row['feature_2']}" for _, row in pairs_df.iterrows()
            ],
            "feature_type": ["interaction"] * len(pairs_df),
            "origin": ["model_factors"] * len(pairs_df),
        }
    )


def generate_nonlinear_transforms(
    X: pd.DataFrame,
    feature_names: list[str],
    strategy: Literal["all", "safe", "none"] = "safe",
) -> pd.DataFrame:
    """Generate nonlinear transform candidates.

    Strategies:
        all: Add all 4 transforms (inverse, log, quadratic, sqrt) for every feature
        safe: Add transforms only for features with appropriate domains
        none: No nonlinear transforms

    Safe filters:
        - inverse: feature has no zeros
        - log: feature is strictly positive
        - quadratic: always safe
        - sqrt: feature is non-negative

    Note: Skips features that already have transform suffixes to avoid duplicates.
    """
    if strategy == "none":
        return pd.DataFrame(columns=["feature_name", "feature_type", "origin"])

    special_cols = {"sample_id", "scenario", "run_id", "AFSC", "UAEORO"}
    X_features = X[[col for col in X.columns if col not in special_cols]]

    # Deduplicate feature_names in case of input error
    unique_features = list(dict.fromkeys(feature_names))  # preserves order
    if len(unique_features) < len(feature_names):
        print(
            "  Warning: Removed "
            f"{len(feature_names) - len(unique_features)} duplicate feature names"
        )

    # Filter out features that already have transform suffixes
    transform_suffixes = ("_inverse", "_log", "_quadratic", "_sqrt")
    base_features = [
        f for f in unique_features if not any(f.endswith(suffix) for suffix in transform_suffixes)
    ]

    if len(base_features) < len(unique_features):
        n_filtered = len(unique_features) - len(base_features)
        print(f"  Note: Skipped {n_filtered} features with transform suffixes")

    transforms = []

    for feature in base_features:
        if feature not in X_features.columns:
            continue

        values = X_features[feature].values

        # Quadratic: always safe
        transforms.append(
            {
                "feature_name": f"{feature}_quadratic",
                "feature_type": "nonlinear",
                "origin": "model_factors",
            }
        )

        if strategy == "all":
            # Add all transforms without checking
            for suffix in ["inverse", "log", "sqrt"]:
                transforms.append(
                    {
                        "feature_name": f"{feature}_{suffix}",
                        "feature_type": "nonlinear",
                        "origin": "model_factors",
                    }
                )
        else:  # strategy == "safe"
            # Inverse: no zeros
            if not np.any(np.abs(values) < 1e-10):
                transforms.append(
                    {
                        "feature_name": f"{feature}_inverse",
                        "feature_type": "nonlinear",
                        "origin": "model_factors",
                    }
                )

            # Log: strictly positive
            if np.all(values > 1e-10):
                transforms.append(
                    {
                        "feature_name": f"{feature}_log",
                        "feature_type": "nonlinear",
                        "origin": "model_factors",
                    }
                )

            # Sqrt: non-negative
            if np.all(values >= -1e-10):
                transforms.append(
                    {
                        "feature_name": f"{feature}_sqrt",
                        "feature_type": "nonlinear",
                        "origin": "model_factors",
                    }
                )

    return pd.DataFrame(transforms)


def validate_catalog(catalog: pd.DataFrame) -> None:
    """Validate catalog structure and contents."""
    required_cols = {"feature_name", "feature_type", "origin"}
    if not required_cols.issubset(catalog.columns):
        raise ValueError(f"Catalog must have columns: {required_cols}")

    valid_types = {"numeric", "interaction", "nonlinear"}
    invalid_types = set(catalog["feature_type"]) - valid_types
    if invalid_types:
        raise ValueError(
            f"Invalid feature_type values: {invalid_types}. Must be one of: {valid_types}"
        )

    # Check for duplicate feature names
    duplicates = catalog["feature_name"].duplicated()
    if duplicates.any():
        dup_names = catalog.loc[duplicates, "feature_name"].unique()
        raise ValueError(
            f"Catalog contains {len(dup_names)} duplicate feature names: {list(dup_names[:5])}"
        )

    # Check interaction format
    interactions = catalog[catalog["feature_type"] == "interaction"]
    for feature_name in interactions["feature_name"]:
        if ":" not in feature_name:
            raise ValueError(
                "Interaction feature "
                f"'{feature_name}' must use colon-delimited format: 'feature1:feature2'"
            )
        parts = feature_name.split(":")
        if len(parts) != 2:
            raise ValueError(f"Interaction feature '{feature_name}' must have exactly 2 factors")

    print("\n✓ Catalog validation passed")


def print_catalog_summary(catalog: pd.DataFrame) -> None:
    """Print summary statistics for generated catalog."""
    type_counts = catalog["feature_type"].value_counts()

    print("\n" + "=" * 70)
    print("FEATURE CATALOG SUMMARY")
    print("=" * 70)
    print(f"Total features: {len(catalog):,}")
    print()

    for feature_type in ["numeric", "interaction", "nonlinear"]:
        count = type_counts.get(feature_type, 0)
        print(f"  {feature_type:12s}: {count:>8,}")

    print()

    # Show sample features
    for feature_type in ["numeric", "interaction", "nonlinear"]:
        subset = catalog[catalog["feature_type"] == feature_type]
        if len(subset) > 0:
            print(f"{feature_type.capitalize()} examples (showing {min(3, len(subset))}):")
            for name in subset["feature_name"].head(3):
                print(f"  - {name}")
            print()

    # Complexity warnings
    n_numeric = type_counts.get("numeric", 0)
    n_interaction = type_counts.get("interaction", 0)
    if n_interaction > 10000:
        print("⚠️  WARNING: Large interaction catalog (>10,000 pairs)")
        print("   Workflow runtime will be significantly longer (20-60+ minutes)")
        print("   Consider using --max-interactions 500-1000 for faster runtime")
        print()

    if n_numeric > 500:
        print("⚠️  WARNING: Large feature set (>500 first-order features)")
        print("   Consider domain-based filtering to reduce dimensionality")
        print()

    print("=" * 70)


def main() -> None:
    """Run catalog generation from command line."""
    parser = argparse.ArgumentParser(
        description="Generate feature catalog for manuscript reproduction workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1].split("Notes:")[0],
    )

    # Required arguments
    parser.add_argument(
        "--input-matrix",
        type=Path,
        required=True,
        help="Path to input matrix (X.parquet) with sample_id column",
    )
    parser.add_argument(
        "--output-catalog",
        type=Path,
        required=True,
        help="Path where catalog should be written (.parquet)",
    )

    # Interaction strategy
    parser.add_argument(
        "--interaction-strategy",
        choices=["all", "top-shap", "from-file", "none"],
        default="top-shap",
        help=(
            "How to select interaction candidates. "
            "'all': all pairwise (WARNING: very large for >100 features), "
            "'top-shap': use SHAP scores to select top N pairs (recommended), "
            "'from-file': load pairs from CSV, "
            "'none': no interactions (workflow discovers dynamically)"
        ),
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=500,
        help="Maximum interaction pairs to include (for top-shap strategy)",
    )
    parser.add_argument(
        "--interaction-file",
        type=Path,
        help=(
            "CSV file with interaction pairs (for from-file strategy). "
            "Must have columns: feature_1, feature_2"
        ),
    )

    # Output matrix (required for SHAP)
    parser.add_argument(
        "--output-matrix",
        type=Path,
        help="Path to output matrix (Y.parquet). Required for --interaction-strategy top-shap",
    )

    # Nonlinear transforms
    parser.add_argument(
        "--nonlinear-strategy",
        choices=["all", "safe", "none"],
        default="safe",
        help=(
            "Which nonlinear transforms to add. "
            "'all': add inverse, log, quadratic, sqrt for every feature, "
            "'safe': add only transforms with valid domains (recommended), "
            "'none': no nonlinear transforms"
        ),
    )

    # Options
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for SHAP analysis",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.input_matrix.exists():
        print(f"ERROR: Input matrix not found: {args.input_matrix}")
        sys.exit(1)

    if args.interaction_strategy == "top-shap" and args.output_matrix is None:
        print("ERROR: --output-matrix required for --interaction-strategy top-shap")
        sys.exit(1)

    if args.interaction_strategy == "from-file" and args.interaction_file is None:
        print("ERROR: --interaction-file required for --interaction-strategy from-file")
        sys.exit(1)

    if args.output_matrix and not args.output_matrix.exists():
        print(f"ERROR: Output matrix not found: {args.output_matrix}")
        sys.exit(1)

    # Create output directory
    args.output_catalog.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FEATURE CATALOG GENERATOR")
    print("=" * 70)
    print(f"Input matrix: {args.input_matrix}")
    print(f"Output catalog: {args.output_catalog}")
    print(f"Interaction strategy: {args.interaction_strategy}")
    print(f"Nonlinear strategy: {args.nonlinear_strategy}")
    print()

    # Load input matrix
    X, feature_names = load_input_matrix(args.input_matrix)

    # Generate catalog components
    catalog_parts = []

    # 1. First-order features
    print("\nGenerating first-order features...")
    first_order = generate_first_order_features(feature_names)
    catalog_parts.append(first_order)
    print(f"  ✓ Added {len(first_order)} first-order features")

    # 2. Interactions
    if args.interaction_strategy == "all":
        print("\nGenerating all pairwise interactions...")
        interactions = generate_all_interactions(feature_names)
        catalog_parts.append(interactions)
        print(f"  ✓ Added {len(interactions)} interaction pairs")

    elif args.interaction_strategy == "top-shap":
        print(f"\nGenerating top {args.max_interactions} SHAP-ranked interactions...")
        Y = pd.read_parquet(args.output_matrix)
        interactions = generate_top_shap_interactions(
            X, Y, feature_names, args.max_interactions, args.random_seed
        )
        catalog_parts.append(interactions)
        print(f"  ✓ Added {len(interactions)} interaction pairs")

    elif args.interaction_strategy == "from-file":
        print(f"\nLoading interactions from {args.interaction_file}...")
        interactions = generate_interactions_from_file(args.interaction_file)
        catalog_parts.append(interactions)
        print(f"  ✓ Added {len(interactions)} interaction pairs")

    else:  # none
        print("\nSkipping interaction generation (strategy=none)")

    # 3. Nonlinear transforms
    if args.nonlinear_strategy != "none":
        print(f"\nGenerating nonlinear transforms (strategy={args.nonlinear_strategy})...")
        nonlinear = generate_nonlinear_transforms(X, feature_names, args.nonlinear_strategy)
        catalog_parts.append(nonlinear)
        print(f"  ✓ Added {len(nonlinear)} nonlinear transforms")
    else:
        print("\nSkipping nonlinear generation (strategy=none)")

    # Combine and validate
    print("\nCombining catalog components...")
    catalog = pd.concat(catalog_parts, ignore_index=True)

    validate_catalog(catalog)

    # Write catalog
    catalog.to_parquet(args.output_catalog, index=False)
    print(f"\n✓ Wrote catalog to: {args.output_catalog}")

    # Print summary
    print_catalog_summary(catalog)

    print("\nNext steps:")
    print("  1. Create dataset config referencing this catalog:")
    print(f"     manuscript_feature_catalog: {args.output_catalog}")
    print("  2. Run workflow:")
    print("     pixi run manuscript-reproduce --config your_config.yml")


if __name__ == "__main__":
    main()
