"""Tests for the Phase 0 manuscript contract freeze layer."""

from __future__ import annotations

from pathlib import Path


def test_manuscript_contract_doc_exists_and_freezes_key_counts() -> None:
    text = Path("docs/manuscript_contract.md").read_text(encoding="utf-8")

    expected_fragments = [
        "160 total = 158 scalar + 2 boolean",
        "635 annual series, 2015--2051",
        "23,495 scalar outputs",
        "26,560",
        "39",
        "90%",
        "349",
        "367",
        "112",
        "37",
        "340",
        "0.0445",
    ]

    for fragment in expected_fragments:
        assert fragment in text


def test_manuscript_contract_doc_freezes_resolved_phase0_choices() -> None:
    text = Path("docs/manuscript_contract.md").read_text(encoding="utf-8")

    expected_fragments = [
        "holdout seed",
        "123",
        "variance floor `epsilon_var`",
        "1e-12",
        "permutation count `B`",
        "200",
        "Jaccard threshold",
        "0.75",
        "Spearman threshold",
        "0.90",
        "95% HC3 Wald intervals",
        (
            "macro average of per-output RMSE divided by the columnwise range of "
            "**training responses**"
        ),
        "These exit criteria are now satisfied.",
    ]

    for fragment in expected_fragments:
        assert fragment in text


def test_manuscript_case_study_config_exists_and_tracks_frozen_items() -> None:
    text = Path("configs/manuscript_case_study.yml").read_text(encoding="utf-8")

    expected_fragments = [
        "contract_status: phase0_complete",
        "n_exogenous_inputs: 160",
        "n_time_series_outputs: 635",
        "n_scalar_outputs: 23495",
        "reported_candidate_count: 26560",
        "holdout_random_seed: 123",
        "epsilon_var: 1.0e-12",
        "epsilon_snr: 1.0e-2",
        "delta: 1.0e-12",
        "permutation_count_B: 200",
        "retained_components: 39",
        "retained_variance_fraction: 0.90",
        "bh_q_screen: 0.10",
        "retained_terms: 349",
        "retained_pairs: 367",
        "identified_transformations: 112",
        "final_support_transformations: 37",
        "ebic_gamma: 0.5",
        "jaccard_threshold: 0.75",
        "spearman_threshold: 0.90",
        "final_predictor_count: 340",
        "final_ols_holdout_nrmse: 0.0445",
        "phase0_blockers: []",
    ]

    for fragment in expected_fragments:
        assert fragment in text


def test_docs_index_includes_manuscript_contract_and_required_guides() -> None:
    text = Path("docs/index.md").read_text(encoding="utf-8")
    assert "manuscript_contract" in text
    assert "reproducibility_example" in text
    assert "scope_boundary" in text
