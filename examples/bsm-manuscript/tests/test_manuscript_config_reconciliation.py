"""Keep the runtime configuration free of derived manuscript result targets."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.reproduce_artifacts import MANUSCRIPT_FIGURES, validate_manuscript_figure_pdfs

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "manuscript_case_study.yml"
FIGURE_DATA = REPO_ROOT / "artifacts" / "figure_data"
ARTIFACTS = REPO_ROOT / "artifacts"


def _minimal_pdf_bytes() -> bytes:
    content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    content += b"x" * (1025 - len(content) - len(b"\nstartxref\n0\n%%EOF\n"))
    return content + b"\nstartxref\n0\n%%EOF\n"


@pytest.fixture(scope="module")
def case_study() -> dict:
    with CONFIG.open() as fh:
        return yaml.safe_load(fh)["case_study"]


@pytest.fixture(scope="module")
def artifact_values() -> dict[str, float | int]:
    summary = pd.read_csv(ARTIFACTS / "final_model" / "final_ols_summary.csv").iloc[0]
    ablation = pd.read_csv(ARTIFACTS / "tables" / "ablation_table.csv").set_index("model_name")
    stages = pd.read_csv(ARTIFACTS / "tables" / "workflow_stage_summary.csv").set_index(
        ["stage", "primary_quantity"]
    )["recomputed_value"]
    return {
        "final_predictor_count": int(summary["n_final_features"]),
        "final_main_effect_count": 63,
        "final_interaction_count": 159,
        "final_transformation_count": 23,
        "intermediate_penalized_holdout_nrmse": float(ablation.loc["penalized_ols", "nrmse"]),
        "final_ols_holdout_nrmse": float(summary["final_ols_holdout_nrmse"]),
        "final_ols_holdout_nrmse_ci_lower": float(summary["final_ols_holdout_nrmse_ci_lower"]),
        "final_ols_holdout_nrmse_ci_upper": float(summary["final_ols_holdout_nrmse_ci_upper"]),
        "retained_terms": int(stages[("empirical_null_screening", "retained_terms")]),
        "retained_pairs": int(stages[("interaction_discovery", "retained_pairs")]),
        "identified_transformations": int(
            stages[("nonlinear_discovery", "retained_transformations")]
        ),
        "source_selected_feature_count": int(
            stages[("sparse_selection_and_stability", "final_stable_support_terms")]
        ),
    }


def test_runtime_config_contains_only_controls(case_study):
    assert set(case_study["final_model"]) == {
        "nrmse_denominator_definition",
        "nrmse_reference_matrix",
        "nrmse_definition_status",
    }
    assert "retained_terms" not in case_study["empirical_null_screen"]
    assert "retained_pairs" not in case_study["interaction_discovery"]
    assert "identified_transformations" not in case_study["nonlinear_discovery"]
    assert "final_support_transformations" not in case_study["nonlinear_discovery"]
    assert "source_selected_feature_count_reference" not in case_study["sparse_selection"]


def test_final_model_counts_come_from_artifacts(artifact_values):
    assert artifact_values["final_predictor_count"] == 245
    assert artifact_values["final_main_effect_count"] == 63
    assert artifact_values["final_interaction_count"] == 159
    assert artifact_values["final_transformation_count"] == 23
    assert (
        artifact_values["final_main_effect_count"]
        + artifact_values["final_interaction_count"]
        + artifact_values["final_transformation_count"]
        == artifact_values["final_predictor_count"]
    )


def test_final_model_nrmse_comes_from_artifacts(artifact_values):
    assert artifact_values["intermediate_penalized_holdout_nrmse"] == pytest.approx(
        0.0682, abs=5e-5
    )
    assert artifact_values["final_ols_holdout_nrmse"] == pytest.approx(0.0679, abs=5e-5)
    assert artifact_values["final_ols_holdout_nrmse_ci_lower"] == pytest.approx(
        0.0663, abs=5e-5
    )
    assert artifact_values["final_ols_holdout_nrmse_ci_upper"] == pytest.approx(
        0.0690, abs=5e-5
    )


def test_stage_counts_come_from_artifacts(case_study, artifact_values):
    assert (
        case_study["output_conditioning"]["temporary_reduction"]["retained_components"]
        == 17
    )
    assert artifact_values["retained_terms"] == 70
    assert artifact_values["retained_pairs"] == 272
    assert artifact_values["identified_transformations"] == 25
    assert artifact_values["source_selected_feature_count"] == 360


def test_release_figure_validator_rejects_missing_or_truncated_pdfs(tmp_path):
    for name in MANUSCRIPT_FIGURES[:-1]:
        (tmp_path / name).write_bytes(b"x" * 1025)

    with pytest.raises(ValueError, match=MANUSCRIPT_FIGURES[-1]):
        validate_manuscript_figure_pdfs(tmp_path)

    (tmp_path / MANUSCRIPT_FIGURES[-1]).write_bytes(b"x" * 1025)
    with pytest.raises(ValueError, match="not valid PDFs"):
        validate_manuscript_figure_pdfs(tmp_path)

    (tmp_path / MANUSCRIPT_FIGURES[-1]).write_bytes(b"x" * 1024)
    with pytest.raises(ValueError, match="empty/truncated"):
        validate_manuscript_figure_pdfs(tmp_path)


def test_release_figure_validator_accepts_complete_pdfs(tmp_path):
    for name in MANUSCRIPT_FIGURES:
        (tmp_path / name).write_bytes(_minimal_pdf_bytes())

    validate_manuscript_figure_pdfs(tmp_path)


def test_support_composition_figure_data_matches_artifact_counts(artifact_values):
    path = FIGURE_DATA / "figure_support_composition_data.csv"
    assert path.exists(), "missing figure_support_composition_data.csv"
    with path.open() as fh:
        rows = {row["feature_type"]: int(row["n_features"]) for row in csv.DictReader(fh)}
    assert rows.get("First Order") == artifact_values["final_main_effect_count"]
    assert rows.get("Second Order") == artifact_values["final_interaction_count"]
    assert rows.get("Non-Linear") == artifact_values["final_transformation_count"]
    assert sum(rows.values()) == artifact_values["final_predictor_count"]


def test_selected_by_module_figure_data_sums_to_artifact_support(artifact_values):
    path = FIGURE_DATA / "figure_selected_by_module_data.csv"
    assert path.exists(), "missing figure_selected_by_module_data.csv"
    with path.open() as fh:
        total = sum(int(row["n_selected_inputs"]) for row in csv.DictReader(fh))
    assert total == artifact_values["final_predictor_count"]
