"""Tests for empirical-null screening provenance artifacts."""

from __future__ import annotations

import pandas as pd

from bsm_rfm.manuscript_stages import (
    EmpiricalNullScreeningResult,
    EmpiricalNullScreeningSpec,
    empirical_null_screening_spec_from_case_study_config,
    write_empirical_null_screening_artifacts,
)


def test_empirical_null_spec_records_public_surrogate_provenance() -> None:
    config = {
        "case_study": {
            "interface": {"holdout_random_seed": 123},
            "empirical_null_screen": {
                "statistic": "coefficient_row_l2_norm",
                "public_implementation_method": "coefficient_row_l2_permutation",
                "public_implementation_status": "source_backed_public_surrogate",
                "source_script_reference": "private_delta_null_screening_script",
                "source_script_equivalence_status": "not_yet_validated",
                "permutation_count_B": 200,
                "bh_q_screen": 0.10,
                "retained_terms": 349,
            },
        }
    }

    spec = empirical_null_screening_spec_from_case_study_config(config)

    assert spec.statistic == "coefficient_row_l2_norm"
    assert spec.implementation_method == "coefficient_row_l2_permutation"
    assert spec.implementation_status == "source_backed_public_surrogate"
    assert spec.source_script_reference == "private_delta_null_screening_script"
    assert spec.source_script_equivalence_status == "not_yet_validated"


def test_empirical_null_spec_uses_public_surrogate_defaults() -> None:
    spec = EmpiricalNullScreeningSpec(
        statistic="coefficient_row_l2_norm",
        permutation_count_B=49,
        bh_q_screen=0.20,
        retained_terms_reference=349,
        random_seed=123,
    )

    assert spec.implementation_method == "coefficient_row_l2_permutation"
    assert spec.implementation_status == "source_backed_public_surrogate"
    assert spec.source_script_reference == "private_delta_null_screening_script"
    assert spec.source_script_equivalence_status == "not_yet_validated"


def test_empirical_null_writer_exports_provenance_table(tmp_path) -> None:
    provenance = pd.DataFrame(
        [
            {
                "stage": "empirical_null_screening",
                "public_implementation_method": "coefficient_row_l2_permutation",
                "public_implementation_status": "source_backed_public_surrogate",
                "source_script_equivalence_status": "not_yet_validated",
            }
        ]
    )
    result = EmpiricalNullScreeningResult(
        feature_screening_statistics=pd.DataFrame([{"feature_name": "x1", "retained": True}]),
        component_coefficients=pd.DataFrame(
            [{"feature_name": "x1", "component": "PC1", "standardized_coefficient": 1.0}]
        ),
        permutation_null_summary=pd.DataFrame([{"feature_name": "x1", "null_mean_statistic": 0.1}]),
        retained_terms=pd.DataFrame([{"feature_name": "x1"}]),
        provenance=provenance,
        summary=pd.DataFrame([{"stage": "empirical_null_screening"}]),
    )

    paths = write_empirical_null_screening_artifacts(result, tmp_path)

    assert "empirical_null_provenance" in paths
    written = pd.read_csv(paths["empirical_null_provenance"])
    assert written.loc[0, "public_implementation_method"] == "coefficient_row_l2_permutation"
    assert written.loc[0, "source_script_equivalence_status"] == "not_yet_validated"
