"""Tests for manuscript empirical-null screening stage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bsm_rfm.manuscript_runtime import (
    build_manuscript_notebook_context,
    load_manuscript_case_study_config,
)
from bsm_rfm.manuscript_stages import (
    EmpiricalNullScreeningSpec,
    build_manuscript_feature_design,
    empirical_null_screening_spec_from_case_study_config,
    run_empirical_null_screening_stage,
    screen_manuscript_empirical_null_terms,
    write_empirical_null_screening_artifacts,
)


def test_empirical_null_screening_spec_matches_frozen_case_study_contract() -> None:
    config = load_manuscript_case_study_config(Path.cwd())
    spec = empirical_null_screening_spec_from_case_study_config(config)

    assert spec.statistic == "coefficient_row_l2_norm"
    assert spec.permutation_count_B == 200
    assert spec.bh_q_screen == 0.10
    assert spec.retained_terms_reference == 349
    assert spec.random_seed == 123


def test_build_manuscript_feature_design_materializes_catalog_terms() -> None:
    inputs = pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "x1": [1.0, 2.0, 3.0],
            "x2": [2.0, 3.0, 4.0],
        }
    )
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x1:x2", "x1_squared", "log1p_x2"],
            "feature_type": ["first_order", "interaction", "transformation", "transformation"],
        }
    )

    design = build_manuscript_feature_design(inputs, catalog)

    assert design["sample_id"].tolist() == [1, 2, 3]
    assert design["x1"].tolist() == [1.0, 2.0, 3.0]
    assert design["x1:x2"].tolist() == [2.0, 6.0, 12.0]
    assert design["x1_squared"].tolist() == [1.0, 4.0, 9.0]
    assert design["log1p_x2"].round(8).tolist() == [1.09861229, 1.38629436, 1.60943791]


def test_empirical_null_screening_retains_train_signal_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    sample_ids = list(range(1, 21))
    x_signal = [float(value) for value in range(20)]
    x_noise = [0.0, 3.0, 1.0, 4.0, 2.0] * 4
    inputs = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "x_signal": x_signal,
            "x_noise": x_noise,
        }
    )
    catalog = pd.DataFrame(
        {
            "feature_name": ["x_signal", "x_noise", "x_signal:x_noise"],
            "feature_type": ["first_order", "first_order", "interaction"],
        }
    )
    holdout = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "split": ["train"] * 16 + ["holdout"] * 4,
        }
    )
    pca_scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "PC1": [2.0 * value for value in x_signal],
            "PC2": [0.25 * value for value in x_signal],
        }
    )
    spec = EmpiricalNullScreeningSpec(
        statistic="coefficient_row_l2_norm",
        permutation_count_B=49,
        bh_q_screen=0.20,
        retained_terms_reference=349,
        random_seed=123,
    )

    result = screen_manuscript_empirical_null_terms(
        inputs,
        catalog,
        holdout,
        pca_scores,
        spec,
    )
    retained = set(result.retained_terms["feature_name"])

    assert "x_signal" in retained
    assert result.summary.loc[0, "n_training_rows"] == 16
    assert result.summary.loc[0, "n_candidate_terms"] == 2
    assert result.summary.loc[0, "n_permutations"] == 49
    assert result.summary.loc[0, "manuscript_retained_terms_reference"] == 349

    paths = write_empirical_null_screening_artifacts(result, tmp_path)
    assert sorted(paths) == [
        "component_coefficients",
        "empirical_null_provenance",
        "empirical_null_screen_summary",
        "feature_screening_statistics",
        "permutation_null_summary",
        "retained_terms",
    ]
    assert paths["retained_terms"].read_text(encoding="utf-8").startswith("feature_name")
    provenance_text = paths["empirical_null_provenance"].read_text(encoding="utf-8")
    assert "source_backed_public_surrogate" in provenance_text
    assert "not_yet_validated" in provenance_text


def test_empirical_null_screening_uses_only_first_order_terms() -> None:
    sample_ids = list(range(1, 21))
    x_signal = [float(value) for value in range(20)]
    x_noise = [0.0, 3.0, 1.0, 4.0, 2.0] * 4
    inputs = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "x_signal": x_signal,
            "x_noise": x_noise,
        }
    )
    catalog = pd.DataFrame(
        {
            "feature_name": ["x_signal", "x_noise", "x_signal:x_noise"],
            "feature_type": ["first_order", "first_order", "interaction"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 16 + ["holdout"] * 4})
    pca_scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "PC1": [2.0 * value for value in x_signal],
            "PC2": [0.25 * value for value in x_signal],
        }
    )
    spec = EmpiricalNullScreeningSpec(
        statistic="coefficient_row_l2_norm",
        permutation_count_B=19,
        bh_q_screen=0.20,
        retained_terms_reference=349,
        random_seed=123,
    )

    result = screen_manuscript_empirical_null_terms(
        inputs,
        catalog,
        holdout,
        pca_scores,
        spec,
    )

    screened_names = set(result.feature_screening_statistics["feature_name"])
    assert screened_names == {"x_signal", "x_noise"}


def test_empirical_null_screening_applies_max_retained_terms_cap() -> None:
    sample_ids = list(range(1, 25))
    x1 = [float(v) for v in range(24)]
    x2 = [float(v) * 0.5 for v in range(24)]
    x3 = [float((v % 5) - 2) for v in range(24)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2, "x3": x3})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order", "first_order", "first_order"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 20 + ["holdout"] * 4})
    pca_scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "PC1": [2.0 * value for value in x1],
            "PC2": [1.5 * value for value in x2],
        }
    )
    spec = EmpiricalNullScreeningSpec(
        statistic="coefficient_row_l2_norm",
        permutation_count_B=19,
        bh_q_screen=1.0,
        retained_terms_reference=349,
        random_seed=123,
        max_retained_terms=1,
    )

    result = screen_manuscript_empirical_null_terms(inputs, catalog, holdout, pca_scores, spec)

    assert len(result.retained_terms) == 1
    assert int(result.summary.loc[0, "n_retained_terms"]) == 1
    assert int(result.feature_screening_statistics["retained"].sum()) == 1


def test_run_empirical_null_screening_stage_executes_demo_context() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(),
        "03_empirical_null_screen.ipynb",
    )

    result = run_empirical_null_screening_stage(context)

    assert result.screening.summary.loc[0, "stage"] == "empirical_null_screening"
    assert result.screening.summary.loc[0, "n_candidate_terms"] == 4
    assert result.artifact_paths["retained_terms"].exists()
