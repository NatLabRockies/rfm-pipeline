"""Fail-closed contracts for the final G11 campaign controller."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.g11_campaign_workflow import (
    _reconcile_accepted_resource_freeze,
    _sacct_array_command,
    _stable_hash,
    _validate_phase_manifest_authorization_paths,
    augment_resource_freeze_accounting,
    adopt_generation_12_prerequisites,
    adopt_gate_b_decision_for_downstream,
    build_campaign_budget_certificate,
    build_development_admission_guard,
    build_downstream_budget_certificate,
    build_publication_contract_amendment,
    main,
    prepare_development_package,
    prepare_downstream_package,
    prepare_final_package,
    submit_authorized_phase,
    validate_completed_submission_record,
    validate_phase_budget_guard,
    validate_scheduler_completion,
    validate_scheduler_completion_accounting,
    write_gate_b_adoption,
)


def test_sacct_array_command_requests_canonical_array_task_ids() -> None:
    assert _sacct_array_command(
        {"worker": "101", "audit": "102"},
        fields=("JobID", "State", "ExitCode", "ElapsedRaw", "AllocNodes"),
    ) == [
        "sacct",
        "--array",
        "-j",
        "101,102",
        "-nP",
        "--format=JobID,State,ExitCode,ElapsedRaw,AllocNodes",
    ]


def test_phase_preflight_rejects_manifest_authorization_path_drift(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "phase": "development",
                "execution_authorization_path": str(
                    tmp_path / "g11-final" / "development.json"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dag = SimpleNamespace(stages=(SimpleNamespace(manifest_path=manifest),))
    expected = tmp_path / "g11-final" / "development" / "authorization.json"

    with pytest.raises(ValueError, match="authorization path differs"):
        _validate_phase_manifest_authorization_paths(
            dag, phase="development", authorization_path=expected
        )

    manifest.write_text(
        json.dumps(
            {
                "phase": "development",
                "execution_authorization_path": str(expected),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        _validate_phase_manifest_authorization_paths(
            dag, phase="development", authorization_path=expected
        )
        == 1
    )


def test_fixed_family_preflight_accepts_confirmatory_manifest_phase_alias(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "g11-final" / "gate_b.json"
    manifest = tmp_path / "fixed-family-manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "phase": "gate_b",
                "stage": "fixed_family_supplement",
                "operation": "fixed_family_supplement",
                "execution_authorization_path": str(expected),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dag = SimpleNamespace(stages=(SimpleNamespace(manifest_path=manifest),))

    assert (
        _validate_phase_manifest_authorization_paths(
            dag,
            phase="fixed_family",
            authorization_path=expected,
        )
        == 1
    )


def test_accepted_freeze_reconciliation_allows_only_accounting_rule_text() -> None:
    accepted_identity = {
        "status": "ACCEPTED",
        "source_hash": "a" * 64,
        "allocation_accounting": {
            "pilot_accounting_sha256": "b" * 64,
            "accepted_pilot_observed_au": 90.0,
            "prior_rejected_attempt_au": 192.55,
            "spent_through_pilot_au": 282.55,
            "accounting_rule": "accepted wording",
        },
    }
    accepted = {
        **accepted_identity,
        "resource_freeze_sha256": _stable_hash(accepted_identity),
    }
    observed_identity = json.loads(json.dumps(accepted_identity))
    observed_identity["allocation_accounting"]["accounting_rule"] = "new wording"
    observed = {
        **observed_identity,
        "resource_freeze_sha256": _stable_hash(observed_identity),
    }

    assert _reconcile_accepted_resource_freeze(observed, accepted) == accepted
    observed_identity["allocation_accounting"]["prior_rejected_attempt_au"] = 192.75
    observed_identity["allocation_accounting"]["spent_through_pilot_au"] = 282.75
    observed_identity["allocation_accounting"]["accounting_rule"] = "new wording"
    observed = {
        **observed_identity,
        "resource_freeze_sha256": _stable_hash(observed_identity),
    }
    reconciled = _reconcile_accepted_resource_freeze(observed, accepted)
    assert reconciled["allocation_accounting"] == {
        **accepted["allocation_accounting"],
        "prior_rejected_attempt_au": 192.75,
        "spent_through_pilot_au": 282.75,
    }
    assert reconciled["resource_freeze_sha256"] == _stable_hash(
        {
            key: value
            for key, value in reconciled.items()
            if key != "resource_freeze_sha256"
        }
    )

    observed_identity["allocation_accounting"]["prior_rejected_attempt_au"] = 192.0
    observed_identity["allocation_accounting"]["spent_through_pilot_au"] = 282.0
    observed = {
        **observed_identity,
        "resource_freeze_sha256": _stable_hash(observed_identity),
    }
    with pytest.raises(ValueError, match="cannot decrease"):
        _reconcile_accepted_resource_freeze(observed, accepted)

    observed_identity["allocation_accounting"]["prior_rejected_attempt_au"] = 192.55
    observed_identity["allocation_accounting"]["spent_through_pilot_au"] = 282.55
    observed["allocation_accounting"]["accepted_pilot_observed_au"] = 91.0
    changed_identity = {
        key: value for key, value in observed.items() if key != "resource_freeze_sha256"
    }
    observed["resource_freeze_sha256"] = _stable_hash(changed_identity)
    with pytest.raises(ValueError, match="differs from accepted evidence"):
        _reconcile_accepted_resource_freeze(observed, accepted)


def test_completed_submission_record_requires_exact_successful_step_coverage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "submission_job_ids.json"
    path.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "run_id": "g11-final",
                "source_hash": "a" * 64,
                "config_hash": "b" * 64,
                "lock_hash": "c" * 64,
                "submission_job_ids": {"one": "101", "two": "102"},
            }
        ),
        encoding="utf-8",
    )

    observed = validate_completed_submission_record(
        path,
        expected_step_ids={"one", "two"},
        expected_identity={
            "run_id": "g11-final",
            "source_hash": "a" * 64,
            "config_hash": "b" * 64,
            "lock_hash": "c" * 64,
        },
    )
    assert observed == {"one": "101", "two": "102"}

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["submission_job_ids"].pop("two")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly cover"):
        validate_completed_submission_record(
            path,
            expected_step_ids={"one", "two"},
            expected_identity={
                "run_id": "g11-final",
                "source_hash": "a" * 64,
                "config_hash": "b" * 64,
                "lock_hash": "c" * 64,
            },
        )


def test_resource_freeze_binds_accepted_and_rejected_allocation_evidence() -> None:
    freeze = {
        "schema_version": 1,
        "status": "ACCEPTED",
        "source_hash": "a" * 64,
        "config_hash": "b" * 64,
        "lock_hash": "c" * 64,
        "telemetry_sha256": "d" * 64,
        "selections": {},
        "resource_freeze_sha256": "placeholder",
    }
    accounting = {
        "status": "COMPLETE",
        "pilot_accounting_sha256": "e" * 64,
        "observed_total_au": 81.25,
    }

    augmented = augment_resource_freeze_accounting(
        freeze,
        pilot_accounting=accounting,
        prior_sunk_au=114.63055555555556,
    )

    allocation = augmented["allocation_accounting"]
    assert allocation["accepted_pilot_observed_au"] == pytest.approx(81.25)
    assert allocation["prior_rejected_attempt_au"] == pytest.approx(114.63055555555556)
    assert allocation["spent_through_pilot_au"] == pytest.approx(195.88055555555556)
    assert len(augmented["resource_freeze_sha256"]) == 64

    with pytest.raises(ValueError, match="nonnegative"):
        augment_resource_freeze_accounting(
            freeze,
            pilot_accounting=accounting,
            prior_sunk_au=-1,
        )


def test_budget_certificate_adds_rejected_attempts_to_full_campaign_envelope() -> None:
    certificate = build_campaign_budget_certificate(
        run_id="g11-final",
        campaign_requested_au=23_700,
        prior_rejected_attempt_au=114.63055555555556,
        postprocessing_reserved_au=5.0,
        allocation_quota_au=25_000,
        resource_freeze_sha256="f" * 64,
        campaign_inventory_hash="1" * 64,
        contract_amendment_sha256="2" * 64,
        accepted_pilot_observed_au=100.0,
        development_resolution_observed_au=200.0,
        remaining_confirmatory_estimated_au=19_500.0,
        remaining_confirmatory_requested_au_with_shared_reserve=23_400.0,
    )
    assert certificate["status"] == "WITHIN_ALLOCATION"
    assert certificate["whole_campaign_projected_au"] == pytest.approx(
        23_819.630555555557
    )
    assert certificate["postprocessing_reserved_au"] == 5.0
    assert certificate["contract_amendment_sha256"] == "2" * 64

    with pytest.raises(ValueError, match="25,000-AU ceiling"):
        build_campaign_budget_certificate(
            run_id="g11-final",
            campaign_requested_au=24_950,
            prior_rejected_attempt_au=114.63055555555556,
            postprocessing_reserved_au=5.0,
            allocation_quota_au=25_000,
            resource_freeze_sha256="f" * 64,
            campaign_inventory_hash="1" * 64,
            contract_amendment_sha256="2" * 64,
            accepted_pilot_observed_au=100.0,
            development_resolution_observed_au=200.0,
            remaining_confirmatory_estimated_au=20_541.0,
            remaining_confirmatory_requested_au_with_shared_reserve=24_650.0,
        )

    with pytest.raises(ValueError, match="requires exact completed-phase accounting"):
        build_campaign_budget_certificate(
            run_id="g11-final",
            campaign_requested_au=23_700,
            prior_rejected_attempt_au=114.63055555555556,
            postprocessing_reserved_au=5.0,
            allocation_quota_au=25_000,
            resource_freeze_sha256="f" * 64,
            campaign_inventory_hash="1" * 64,
            contract_amendment_sha256="2" * 64,
        )


def test_budget_certificate_exposes_and_enforces_resolution_au_ceiling() -> None:
    accepted_pilot_au = 90.36944444444445
    prior_rejected_au = 192.55
    remaining_requested_au = 24_301
    resolution_limit = (
        25_000 - accepted_pilot_au - prior_rejected_au - remaining_requested_au - 5.0
    )
    certificate = build_campaign_budget_certificate(
        run_id="g11-final",
        campaign_requested_au=(
            accepted_pilot_au + resolution_limit + remaining_requested_au
        ),
        prior_rejected_attempt_au=prior_rejected_au,
        postprocessing_reserved_au=5.0,
        allocation_quota_au=25_000,
        resource_freeze_sha256="f" * 64,
        campaign_inventory_hash="1" * 64,
        contract_amendment_sha256="2" * 64,
        accepted_pilot_observed_au=accepted_pilot_au,
        development_resolution_observed_au=resolution_limit,
        remaining_confirmatory_estimated_au=20_250.751923076525,
        remaining_confirmatory_requested_au_with_shared_reserve=(
            remaining_requested_au
        ),
    )

    assert certificate["development_resolution_au_ceiling_for_selected_design"] == (
        pytest.approx(411.0805555555562)
    )
    assert certificate["whole_campaign_projected_au"] == pytest.approx(25_000)

    with pytest.raises(ValueError, match="25,000-AU ceiling"):
        build_campaign_budget_certificate(
            run_id="g11-final",
            campaign_requested_au=(
                accepted_pilot_au + resolution_limit + 0.01 + remaining_requested_au
            ),
            prior_rejected_attempt_au=prior_rejected_au,
            postprocessing_reserved_au=5.0,
            allocation_quota_au=25_000,
            resource_freeze_sha256="f" * 64,
            campaign_inventory_hash="1" * 64,
            contract_amendment_sha256="2" * 64,
            accepted_pilot_observed_au=accepted_pilot_au,
            development_resolution_observed_au=resolution_limit + 0.01,
            remaining_confirmatory_estimated_au=20_250.751923076525,
            remaining_confirmatory_requested_au_with_shared_reserve=(
                remaining_requested_au
            ),
        )


def test_development_admission_guard_preserves_the_affordable_final_branch() -> None:
    freeze_identity = {
        "allocation_accounting": {
            "accepted_pilot_observed_au": 90.36944444444445,
            "prior_rejected_attempt_au": 192.55,
        },
    }
    freeze = {
        **freeze_identity,
        "resource_freeze_sha256": _stable_hash(freeze_identity),
    }
    stage_allocations = [SimpleNamespace(stage_name="resolution", requested_au=300.0)]

    guard = build_development_admission_guard(
        stage_allocations=stage_allocations,
        resource_freeze=freeze,
        postprocessing_reserved_au=5.0,
        allocation_quota_au=25_000.0,
    )

    assert guard["status"] == "DEVELOPMENT_WITHIN_ALLOCATION"
    assert guard["development_requested_au"] == 300.0
    assert guard["b999_confirmatory_requested_au_with_shared_reserve"] == 24_301
    assert guard["whole_campaign_maximum_au"] == pytest.approx(24_888.919444444444)

    stage_allocations[0] = SimpleNamespace(stage_name="resolution", requested_au=500.0)
    with pytest.raises(ValueError, match="affordable B=999 branch"):
        build_development_admission_guard(
            stage_allocations=stage_allocations,
            resource_freeze=freeze,
            postprocessing_reserved_au=5.0,
            allocation_quota_au=25_000.0,
        )


def test_budget_cli_replaces_completed_stage_estimates_with_actual_au(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    (package / "contract").mkdir(parents=True)
    summary = {
        "run_id": "g11-final",
        "source_hash": "a" * 64,
        "config_hash": "9" * 64,
        "lock_hash": "b" * 64,
        "campaign_inventory_hash": "1" * 64,
        "campaign_envelope": {
            "requested_au": 25_215,
            "stage_allocations": [
                {"stage_name": "pilot_conditioning", "estimated_au": 100.0},
                {"stage_name": "resolution", "estimated_au": 300.0},
                {"stage_name": "gate_b", "estimated_au": 20_000.0},
            ],
        },
    }
    summary_path = package / "package_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    amendment = {
        "contract_amendment_sha256": "2" * 64,
    }
    (package / "contract" / "fixed_family_amendment.json").write_text(
        json.dumps(amendment), encoding="utf-8"
    )
    freeze = {
        "source_hash": "a" * 64,
        "config_hash": "c" * 64,
        "lock_hash": "b" * 64,
        "resource_freeze_sha256": "f" * 64,
        "allocation_accounting": {
            "prior_rejected_attempt_au": 192.55,
            "accepted_pilot_observed_au": 90.36944444444445,
        },
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    completion_identity = {
        "schema_version": 1,
        "status": "PHASE_COMPLETE",
        "phase": "development",
        "run_id": "g11-final",
        "source_hash": "a" * 64,
        "config_hash": "c" * 64,
        "lock_hash": "b" * 64,
        "observed_total_au": 250.0,
    }
    completion_path = tmp_path / "development.json"
    completion_path.write_text(
        json.dumps(
            {
                **completion_identity,
                "completion_sha256": _stable_hash(completion_identity),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "budget.json"

    assert (
        main(
            [
                "certify-budget",
                "--package-summary",
                str(summary_path),
                "--resource-freeze",
                str(freeze_path),
                "--development-completion",
                str(completion_path),
                "--allocation-quota",
                "25000",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    certificate = json.loads(output.read_text(encoding="utf-8"))
    assert certificate["accepted_pilot_observed_au"] == pytest.approx(90.36944444444445)
    assert certificate["development_resolution_observed_au"] == 250.0
    assert certificate["remaining_confirmatory_estimated_au"] == 20_000.0
    assert certificate["remaining_confirmatory_requested_au_with_shared_reserve"] == (
        24_000
    )


def test_phase_budget_guard_blocks_single_phase_from_crossing_cap() -> None:
    allocations = (
        SimpleNamespace(
            stage_name="gate_b", estimated_au=14_418.0, requested_au=18_031.0
        ),
        SimpleNamespace(
            stage_name="applied_bootstrap", estimated_au=522.0, requested_au=652.0
        ),
        SimpleNamespace(
            stage_name="recovery", estimated_au=4_000.0, requested_au=6_639.0
        ),
    )
    initial = validate_phase_budget_guard(
        stage_allocations=allocations,
        completed_stage_names=set(),
        current_stage_names={"gate_b"},
        prior_rejected_attempt_au=192.55,
        accepted_pilot_observed_au=90.36944444444445,
        development_resolution_observed_au=250.0,
        completed_confirmatory_observed_au=0.0,
        postprocessing_reserved_au=5.0,
        allocation_quota_au=25_000.0,
    )
    assert initial["status"] == "PHASE_WITHIN_ALLOCATION"
    assert initial["current_phase_requested_au"] == 18_031.0

    with pytest.raises(ValueError, match="current phase could cross"):
        validate_phase_budget_guard(
            stage_allocations=allocations,
            completed_stage_names={"gate_b", "applied_bootstrap"},
            current_stage_names={"recovery"},
            prior_rejected_attempt_au=192.55,
            accepted_pilot_observed_au=90.36944444444445,
            development_resolution_observed_au=250.0,
            completed_confirmatory_observed_au=18_031.0 + 652.0,
            postprocessing_reserved_au=5.0,
            allocation_quota_au=25_000.0,
        )


def test_publication_amendment_changes_only_fixed_family_replicates() -> None:
    from rfm_pipeline.campaign_contract import G11_CONTRACT

    publication, amendment = build_publication_contract_amendment(G11_CONTRACT)

    before = asdict(G11_CONTRACT)
    after = asdict(publication)
    changed = {key for key in before if before[key] != after[key]}
    assert changed == {"fixed_family_replicates"}
    assert before["fixed_family_replicates"] == 1000
    assert publication.fixed_family_replicates == 300
    assert (
        len([scenario for scenario in publication.scenarios if scenario.kind == "null"])
        == 5
    )
    assert {
        scenario.n_replicates
        for scenario in publication.scenarios
        if scenario.kind == "null"
    } == {1000}
    assert amendment["changed_fields"] == {
        "fixed_family_replicates": {"before": 1000, "after": 300}
    }
    assert amendment["scientific_generation_transition"] == {
        "superseded_schema_version": "g11_campaign_contract_v10",
        "superseded_generation": 11,
        "superseded_contract_hash": (
            "66f9a7fb702c0726c464393f7c153001846510d891a75ccb2c1d48b467375c24"
        ),
        "selected_schema_version": "g11_campaign_contract_v11",
        "selected_generation": 12,
        "changed_scientific_fields": {
            "tree_family_alpha": {"before": 0.025, "after": 0.020},
            "binary_binary_family_alpha": {"before": 0.025, "after": 0.020},
            "fixed_family_replicates": {"before": 200, "after": 300},
        },
        "superseded_gate_b_evidence_status": "development_only_not_adoptable",
        "superseded_fixed_family_evidence_status": ("failed_confirmatory_not_reusable"),
        "required_fresh_phases": ["gate_b", "fixed_family_supplement"],
        "seed_policy": "contract_hash_derived_zero_overlap_required",
    }
    assert amendment["fixed_family_gate"]["largest_passing_false_selection_count"] == 18
    assert amendment["execution_rule"] == (
        "exactly 300 precommitted replicates; no interim testing, optional stopping, "
        "or later incremental expansion"
    )
    assert "approved campaign allocation" in amendment["rationale"]
    assert "25,000" not in amendment["rationale"]
    assert len(amendment["contract_amendment_sha256"]) == 64


def test_generation_12_confirmation_seeds_do_not_overlap_failed_generation() -> None:
    from rfm_pipeline.campaign_contract import (
        G11_CONTRACT,
        compute_contract_hash,
        derive_seed,
    )

    publication, _ = build_publication_contract_amendment(G11_CONTRACT)
    selected_hash = compute_contract_hash(publication)
    failed_hash = "66f9a7fb702c0726c464393f7c153001846510d891a75ccb2c1d48b467375c24"
    gate_b_identities = [
        (scenario.id, replicate_index)
        for scenario in G11_CONTRACT.scenarios
        if scenario.kind in {"null", "strong"}
        for replicate_index in range(scenario.n_replicates)
    ]
    assert len(gate_b_identities) == 5600

    failed_seeds = {
        derive_seed(failed_hash, scenario_id, replicate_index)
        for scenario_id, replicate_index in gate_b_identities
    }
    failed_seeds.update(
        derive_seed(failed_hash, "fixed_family_supplement", replicate_index)
        for replicate_index in range(200)
    )
    selected_seeds = {
        derive_seed(selected_hash, scenario_id, replicate_index)
        for scenario_id, replicate_index in gate_b_identities
    }
    selected_seeds.update(
        derive_seed(selected_hash, "fixed_family_supplement", replicate_index)
        for replicate_index in range(300)
    )

    assert len(failed_seeds) == 5800
    assert len(selected_seeds) == 5900
    assert failed_seeds.isdisjoint(selected_seeds)


def test_generation_12_prerequisite_adoption_preserves_resource_decision() -> None:
    source_freeze = {
        "schema_version": 1,
        "status": "ACCEPTED",
        "selector": "minimum_projected_au_subject_to_resource_bounds",
        "source_hash": "a" * 64,
        "config_hash": "b" * 64,
        "lock_hash": "c" * 64,
        "telemetry_sha256": "d" * 64,
        "selections": {"pilot": {"status": "ACCEPTED", "profile": "p1"}},
        "allocation_accounting": {"spent_through_pilot_au": 401.0},
    }
    source_freeze["resource_freeze_sha256"] = _stable_hash(source_freeze)
    source_resolution = {
        "operation": "resolution",
        "status": "completed",
        "decision": "ACCEPTED",
        "terminal_record_count": 20,
        "selected_B_interaction": 999,
        "contract_hash": "b" * 64,
    }
    source_completion = {
        "schema_version": 1,
        "status": "PHASE_COMPLETE",
        "phase": "development",
        "run_id": "g11-old",
        "source_hash": "a" * 64,
        "config_hash": "b" * 64,
        "lock_hash": "c" * 64,
        "observed_total_au": 0.125,
        "campaign_inventory_hash": "e" * 64,
    }
    source_completion["completion_sha256"] = _stable_hash(source_completion)

    freeze, resolution, completion, adoption = adopt_generation_12_prerequisites(
        source_resource_freeze=source_freeze,
        source_resolution_decision=source_resolution,
        source_resolution_decision_sha256="f" * 64,
        source_development_completion=source_completion,
        target_source_hash="1" * 64,
        target_config_hash="2" * 64,
        target_lock_hash="3" * 64,
        target_run_id="g11-generation-12",
    )

    assert freeze["selections"] == source_freeze["selections"]
    assert freeze["telemetry_sha256"] == source_freeze["telemetry_sha256"]
    assert freeze["allocation_accounting"] == source_freeze["allocation_accounting"]
    assert freeze["source_hash"] == "1" * 64
    assert freeze["config_hash"] == "2" * 64
    assert freeze["lock_hash"] == "3" * 64
    assert freeze["resource_freeze_sha256"] == _stable_hash(
        {key: value for key, value in freeze.items() if key != "resource_freeze_sha256"}
    )
    assert resolution["selected_B_interaction"] == 999
    assert resolution["contract_hash"] == "2" * 64
    assert resolution["adopted_resolution_decision_sha256"] == "f" * 64
    assert completion["run_id"] == "g11-generation-12"
    assert completion["observed_total_au"] == 0.125
    assert completion["completion_sha256"] == _stable_hash(
        {key: value for key, value in completion.items() if key != "completion_sha256"}
    )
    assert adoption["resource_decision_changed"] is False
    assert adoption["scientific_result_reused"] == "resolution_only"
    assert adoption["generation_adoption_sha256"] == _stable_hash(
        {
            key: value
            for key, value in adoption.items()
            if key != "generation_adoption_sha256"
        }
    )


def test_generation_12_prerequisite_adoption_rejects_stale_source() -> None:
    source_freeze = {
        "schema_version": 1,
        "status": "ACCEPTED",
        "source_hash": "a" * 64,
        "config_hash": "b" * 64,
        "lock_hash": "c" * 64,
        "telemetry_sha256": "d" * 64,
        "selections": {},
        "resource_freeze_sha256": "0" * 64,
    }
    source_resolution = {
        "operation": "resolution",
        "status": "completed",
        "decision": "ACCEPTED",
        "terminal_record_count": 20,
        "selected_B_interaction": 999,
        "contract_hash": "b" * 64,
    }
    source_completion = {
        "status": "PHASE_COMPLETE",
        "phase": "development",
        "run_id": "g11-old",
        "source_hash": "a" * 64,
        "config_hash": "b" * 64,
        "lock_hash": "c" * 64,
        "observed_total_au": 0.125,
        "completion_sha256": "0" * 64,
    }

    with pytest.raises(ValueError, match="source resource freeze self-hash differs"):
        adopt_generation_12_prerequisites(
            source_resource_freeze=source_freeze,
            source_resolution_decision=source_resolution,
            source_resolution_decision_sha256="f" * 64,
            source_development_completion=source_completion,
            target_source_hash="1" * 64,
            target_config_hash="2" * 64,
            target_lock_hash="3" * 64,
            target_run_id="g11-generation-12",
        )


def test_gate_b_pass_is_adopted_only_across_fixed_family_count_amendment() -> None:
    from rfm_pipeline.campaign_contract import G11_CONTRACT, compute_contract_hash

    publication, amendment = build_publication_contract_amendment(G11_CONTRACT)
    new_hash = compute_contract_hash(publication)
    amendment_identity = {
        key: value
        for key, value in amendment.items()
        if key != "contract_amendment_sha256"
    }
    amendment_identity["selected_contract_hash"] = new_hash
    amendment = {
        **amendment_identity,
        "contract_amendment_sha256": _stable_hash(amendment_identity),
    }
    original = {
        "operation": "gate_b",
        "status": "completed",
        "decision": "PASS",
        "contract_hash": compute_contract_hash(G11_CONTRACT),
        "terminal_record_count": 5600,
    }

    adopted = adopt_gate_b_decision_for_downstream(
        gate_b_decision=original,
        gate_b_decision_sha256="a" * 64,
        amendment=amendment,
    )

    assert adopted["operation"] == "gate_b"
    assert adopted["decision"] == "PASS"
    assert adopted["contract_hash"] == new_hash
    assert adopted["adopted_from_contract_hash"] == original["contract_hash"]
    assert adopted["adopted_gate_b_decision_sha256"] == "a" * 64
    identity = {
        key: value for key, value in adopted.items() if key != "adoption_sha256"
    }
    assert adopted["adoption_sha256"] == _stable_hash(identity)


def test_gate_b_adoption_writer_hashes_exact_decision_bytes(tmp_path: Path) -> None:
    from rfm_pipeline.campaign_contract import G11_CONTRACT, compute_contract_hash

    publication, amendment = build_publication_contract_amendment(G11_CONTRACT)
    amendment_identity = {
        key: value
        for key, value in amendment.items()
        if key != "contract_amendment_sha256"
    }
    amendment_identity["selected_contract_hash"] = compute_contract_hash(publication)
    amendment = {
        **amendment_identity,
        "contract_amendment_sha256": _stable_hash(amendment_identity),
    }
    amendment_path = tmp_path / "amendment.json"
    amendment_path.write_text(json.dumps(amendment), encoding="utf-8")
    decision = {
        "operation": "gate_b",
        "status": "completed",
        "decision": "PASS",
        "contract_hash": compute_contract_hash(G11_CONTRACT),
        "terminal_record_count": 5600,
    }
    decision_path = tmp_path / "gate_b_decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    output = tmp_path / "gate_b_adoption.json"

    adopted = write_gate_b_adoption(
        gate_b_decision_path=decision_path,
        amendment_path=amendment_path,
        output_path=output,
    )

    import hashlib

    assert (
        adopted["adopted_gate_b_decision_sha256"]
        == hashlib.sha256(decision_path.read_bytes()).hexdigest()
    )
    assert json.loads(output.read_text(encoding="utf-8")) == adopted


def test_downstream_budget_certificate_reserves_only_unexecuted_work() -> None:
    certificate = build_downstream_budget_certificate(
        run_id="g11-final",
        completed_observed_au=5_000.25,
        remaining_downstream_estimated_au=20_000.1,
        postprocessing_reserved_au=5.0,
        allocation_quota_au=30_000.0,
        resource_freeze_sha256="a" * 64,
        campaign_inventory_hash="b" * 64,
        contract_amendment_sha256="c" * 64,
    )

    assert (
        certificate["remaining_downstream_requested_au_with_shared_reserve"] == 24_001
    )
    assert certificate["whole_campaign_projected_au"] == pytest.approx(29_006.25)
    assert certificate["shared_reserve_fraction_on_unexecuted_downstream_work"] == 0.20

    with pytest.raises(ValueError, match="allocation ceiling"):
        build_downstream_budget_certificate(
            run_id="g11-final",
            completed_observed_au=6_000.0,
            remaining_downstream_estimated_au=20_000.1,
            postprocessing_reserved_au=5.0,
            allocation_quota_au=30_000.0,
            resource_freeze_sha256="a" * 64,
            campaign_inventory_hash="b" * 64,
            contract_amendment_sha256="c" * 64,
        )


def test_downstream_package_omits_gate_b_and_binds_exact_completed_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rfm_pipeline.hpc_campaign_package as hpc
    from rfm_pipeline.campaign_contract import (
        G11_CONTRACT,
        compute_contract_hash,
        render_contract_toml,
    )

    repo = tmp_path / "rfm"
    (repo / "src" / "rfm_pipeline").mkdir(parents=True)
    (repo / "pixi.lock").write_text("lock", encoding="utf-8")
    config = tmp_path / "campaign.yml"
    config.write_text("config", encoding="utf-8")
    base_contract = replace(
        G11_CONTRACT,
        B_interaction=999,
        resolution_decision_sha256="f" * 64,
    )
    contract_path = tmp_path / "gate_b_contract.toml"
    contract_path.write_text(
        render_contract_toml(
            base_contract,
            compute_contract_hash(base_contract),
            gate="HPC",
            status="NO_SUBMIT",
        ),
        encoding="utf-8",
    )
    freeze_path = tmp_path / "resource_freeze.json"
    freeze = {
        "status": "ACCEPTED",
        "source_hash": "a" * 64,
        "config_hash": compute_contract_hash(base_contract),
        "lock_hash": "b" * 64,
        "resource_freeze_sha256": "c" * 64,
    }
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    monkeypatch.setattr(hpc, "_validate_resource_freeze", lambda _freeze: None)
    monkeypatch.setattr(hpc, "_hash_python_tree", lambda _path: "a" * 64)
    monkeypatch.setattr(
        hpc,
        "_load_campaign_config",
        lambda _path: {"cluster": {"allocation_quota": "30000"}},
    )

    def fake_hash(path: Path) -> str:
        return "b" * 64 if Path(path).name == "pixi.lock" else "d" * 64

    monkeypatch.setattr(hpc, "_hash_file", fake_hash)
    observed: dict[str, object] = {}

    def fake_generate(**kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        output = Path(str(kwargs["output_dir"]))
        (output / "contract").mkdir(parents=True)
        contract = kwargs["contract"]
        return SimpleNamespace(
            run_id="g11-final",
            config_hash=compute_contract_hash(contract),
            campaign_inventory_hash="e" * 64,
            campaign_envelope=SimpleNamespace(
                stage_allocations=(
                    SimpleNamespace(
                        stage_name="fixed_family_supplement", estimated_au=100.0
                    ),
                    SimpleNamespace(stage_name="recovery", estimated_au=200.0),
                )
            ),
        )

    monkeypatch.setattr(hpc, "generate_campaign_package", fake_generate)
    output = tmp_path / "downstream"
    certificate_path = tmp_path / "downstream_budget.json"

    dag, certificate = prepare_downstream_package(
        output_dir=output,
        resource_freeze_path=freeze_path,
        gate_b_contract_path=contract_path,
        repo_root=repo,
        config_path=config,
        completed_observed_au=500.0,
        allocation_quota_au=30_000.0,
        postprocessing_reserved_au=5.0,
        budget_certificate_path=certificate_path,
    )

    assert observed["package_mode"] == "downstream"
    assert observed["completed_observed_au_for_admission"] == 500.0
    assert observed["postprocessing_reserved_au_for_admission"] == 5.0
    assert observed["contract"].fixed_family_replicates == 300
    assert certificate["remaining_downstream_estimated_au"] == 300.0
    assert certificate["remaining_downstream_requested_au_with_shared_reserve"] == 360
    amendment = json.loads(
        (output / "contract" / "fixed_family_amendment.json").read_text(
            encoding="utf-8"
        )
    )
    assert amendment["selected_contract_hash"] == dag.config_hash
    assert certificate_path.is_file()


def test_development_package_requests_resolution_only_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rfm_pipeline.hpc_campaign_package as hpc

    freeze_path = tmp_path / "resource_freeze.json"
    freeze_path.write_text(
        json.dumps({"status": "ACCEPTED", "allocation_accounting": {"spent_au": 1.0}}),
        encoding="utf-8",
    )
    observed: dict[str, object] = {}
    sentinel = SimpleNamespace(name="development")

    def fake_generate(**kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(hpc, "generate_campaign_package", fake_generate)

    result = prepare_development_package(
        output_dir=tmp_path / "development",
        resource_freeze_path=freeze_path,
        repo_root=tmp_path / "rfm",
        config_path=tmp_path / "campaign.yml",
    )

    assert result is sentinel
    assert observed["package_mode"] == "development"


def test_final_package_uses_amended_contract_but_preserves_pilot_freeze_basis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rfm_pipeline.hpc_campaign_package as hpc
    from rfm_pipeline.campaign_contract import G11_CONTRACT, compute_contract_hash

    repo = tmp_path / "rfm"
    (repo / "src" / "rfm_pipeline").mkdir(parents=True)
    (repo / "pixi.lock").write_text("lock", encoding="utf-8")
    config = tmp_path / "campaign.yml"
    config.write_text("config", encoding="utf-8")
    freeze_path = tmp_path / "resource_freeze.json"
    freeze = {
        "schema_version": 1,
        "status": "ACCEPTED",
        "source_hash": "a" * 64,
        "config_hash": compute_contract_hash(G11_CONTRACT),
        "lock_hash": "b" * 64,
        "telemetry_sha256": "c" * 64,
        "selections": {},
        "allocation_accounting": {
            "prior_rejected_attempt_au": 192.55,
            "accepted_pilot_observed_au": 90.36944444444445,
        },
        "resource_freeze_sha256": "d" * 64,
    }
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    decision_path = tmp_path / "resolution_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "operation": "resolution",
                "status": "completed",
                "decision": "ACCEPTED",
                "terminal_record_count": 20,
                "contract_hash": compute_contract_hash(G11_CONTRACT),
                "selected_B_interaction": 999,
            }
        ),
        encoding="utf-8",
    )
    completion_identity = {
        "schema_version": 1,
        "status": "PHASE_COMPLETE",
        "phase": "development",
        "run_id": "g11-final",
        "source_hash": "a" * 64,
        "config_hash": compute_contract_hash(G11_CONTRACT),
        "lock_hash": "b" * 64,
        "observed_total_au": 250.0,
    }
    completion_path = tmp_path / "development_completion.json"
    completion_path.write_text(
        json.dumps(
            {
                **completion_identity,
                "completion_sha256": _stable_hash(completion_identity),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(hpc, "_validate_resource_freeze", lambda _freeze: None)
    monkeypatch.setattr(hpc, "_hash_python_tree", lambda _path: "a" * 64)
    monkeypatch.setattr(
        hpc, "_load_campaign_config", lambda _path: {"run_id": "g11-final"}
    )

    def fake_hash(path: Path) -> str:
        return "b" * 64 if Path(path).name == "pixi.lock" else "e" * 64

    monkeypatch.setattr(hpc, "_hash_file", fake_hash)
    observed: dict[str, object] = {}

    def fake_generate(**kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        output = Path(str(kwargs["output_dir"]))
        (output / "contract").mkdir(parents=True)
        contract = kwargs["contract"]
        return SimpleNamespace(
            run_id="g11-final",
            output_dir=output,
            config_hash=compute_contract_hash(contract),
            campaign_inventory_hash="1" * 64,
            campaign_envelope=SimpleNamespace(
                requested_au=23_700.0,
                stage_allocations=(
                    SimpleNamespace(
                        stage_name="pilot_conditioning", estimated_au=100.0
                    ),
                    SimpleNamespace(stage_name="resolution", estimated_au=300.0),
                    SimpleNamespace(stage_name="gate_b", estimated_au=20_000.0),
                ),
            ),
        )

    monkeypatch.setattr(hpc, "generate_campaign_package", fake_generate)
    output = tmp_path / "final-package"
    certificate_path = tmp_path / "budget.json"

    dag, certificate = prepare_final_package(
        output_dir=output,
        resource_freeze_path=freeze_path,
        resolution_decision_path=decision_path,
        development_completion_path=completion_path,
        repo_root=repo,
        config_path=config,
        allocation_quota_au=25_000,
        postprocessing_reserved_au=5.0,
        budget_certificate_path=certificate_path,
    )

    selected = observed["contract"]
    assert selected.fixed_family_replicates == 300
    assert selected.B_interaction == 999
    assert selected.resolution_decision_sha256 == "e" * 64
    assert observed["resource_freeze"] == freeze
    assert observed["completed_observed_au_for_admission"] == pytest.approx(
        90.36944444444445 + 192.55 + 250.0
    )
    assert observed["postprocessing_reserved_au_for_admission"] == 5.0
    amendment_path = output / "contract" / "fixed_family_amendment.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    assert amendment["selected_contract_hash"] == dag.config_hash
    assert (
        certificate["contract_amendment_sha256"]
        == amendment["contract_amendment_sha256"]
    )
    assert certificate["accepted_pilot_observed_au"] == pytest.approx(90.36944444444445)
    assert certificate["development_resolution_observed_au"] == 250.0
    assert certificate["remaining_confirmatory_estimated_au"] == 20_000.0
    assert (
        certificate["remaining_confirmatory_requested_au_with_shared_reserve"] == 24_000
    )


def test_scheduler_completion_requires_exact_completed_zero_exit_rows() -> None:
    observed = validate_scheduler_completion(
        {"step-a": "101", "step-b": "102"},
        "101|COMPLETED|0:0\n101.batch|COMPLETED|0:0\n102|COMPLETED|0:0\n",
    )
    assert observed == {
        "step-a": {"job_id": "101", "state": "COMPLETED", "exit_code": "0:0"},
        "step-b": {"job_id": "102", "state": "COMPLETED", "exit_code": "0:0"},
    }

    with pytest.raises(ValueError, match="not exactly COMPLETED/0:0"):
        validate_scheduler_completion(
            {"step-a": "101", "step-b": "102"},
            "101|COMPLETED|0:0\n102|FAILED|1:0\n",
        )

    with pytest.raises(ValueError, match="exact top-level"):
        validate_scheduler_completion(
            {"step-a": "101", "step-b": "102"},
            "101|COMPLETED|0:0\n",
        )


def test_scheduler_completion_accounting_uses_exact_top_level_elapsed_nodes() -> None:
    observed, total_au = validate_scheduler_completion_accounting(
        {"step-a": "101", "step-b": "102"},
        "101|COMPLETED|0:0|3600|1\n"
        "101.batch|COMPLETED|0:0|3600|1\n"
        "102|COMPLETED|0:0|1800|2\n",
        cpu_charge_factor=10.0,
        qos_factor=1.0,
    )

    assert observed["step-a"]["elapsed_seconds"] == 3600
    assert observed["step-b"]["allocated_nodes"] == 2
    assert total_au == pytest.approx(20.0)

    with pytest.raises(ValueError, match="accounting row"):
        validate_scheduler_completion_accounting(
            {"step-a": "101"},
            "101|COMPLETED|0:0|-1|1\n",
            cpu_charge_factor=10.0,
            qos_factor=1.0,
        )


def test_scheduler_completion_accounting_sums_every_array_task_and_shared_fraction() -> (
    None
):
    observed, total_au = validate_scheduler_completion_accounting(
        {"exclusive-array": "101", "shared-array": "102"},
        "101_0|COMPLETED|0:0|3600|1\n"
        "101_0.batch|COMPLETED|0:0|3600|1\n"
        "101_1|COMPLETED|0:0|1800|1\n"
        "102_0|COMPLETED|0:0|3600|1\n"
        "102_1|COMPLETED|0:0|3600|1\n",
        cpu_charge_factor=10.0,
        qos_factor=1.0,
        step_accounting={
            "exclusive-array": {
                "array_task_count": 2,
                "shared_node_equivalent": None,
            },
            "shared-array": {
                "array_task_count": 2,
                "shared_node_equivalent": 0.25,
            },
        },
    )

    assert observed["exclusive-array"]["accounted_task_count"] == 2
    assert observed["shared-array"]["accounted_task_count"] == 2
    assert observed["shared-array"]["observed_au"] == pytest.approx(5.0)
    assert total_au == pytest.approx(20.0)

    with pytest.raises(ValueError, match="array-task coverage"):
        validate_scheduler_completion_accounting(
            {"exclusive-array": "101"},
            "101_0|COMPLETED|0:0|3600|1\n",
            cpu_charge_factor=10.0,
            qos_factor=1.0,
            step_accounting={
                "exclusive-array": {
                    "array_task_count": 2,
                    "shared_node_equivalent": None,
                }
            },
        )


def test_phase_submission_journals_each_job_id_before_later_sbatch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rfm_pipeline.hpc_campaign_package as hpc

    plan = {
        "submission_plan_sha256": "1" * 64,
        "steps": [{"step_id": "one"}, {"step_id": "two"}],
    }
    monkeypatch.setattr(hpc, "build_campaign_phase_plan", lambda *_a, **_k: plan)

    def fail_second(_plan: object, *, run_command: object, **_kwargs: object) -> object:
        first = run_command(["sbatch", "one.sbatch"])
        assert first.stdout == "101\n"
        run_command(["sbatch", "two.sbatch"])
        raise RuntimeError("second submission failed")

    monkeypatch.setattr(hpc, "execute_campaign_phase_plan", fail_second)
    calls = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="101\n", stderr=""),
            subprocess.CompletedProcess([], 1, stdout="", stderr="denied"),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ]
    )
    monkeypatch.setenv("PYTHONPATH", "/tmp/wrong-rfm")
    scheduler_environments: list[dict[str, str]] = []

    def fake_run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        scheduler_environments.append(kwargs["env"])
        return next(calls)

    monkeypatch.setattr(subprocess, "run", fake_run)
    preflight = tmp_path / "preflight.json"
    authorization = tmp_path / "authorization.json"
    preflight.write_text(json.dumps({"preflight_sha256": "2" * 64}), encoding="utf-8")
    authorization.write_text(
        json.dumps({"authorization_sha256": "3" * 64}), encoding="utf-8"
    )
    record = tmp_path / "submission.json"
    dag = SimpleNamespace(
        run_id="g11-final",
        source_hash="a" * 64,
        config_hash="b" * 64,
        lock_hash="c" * 64,
        campaign_inventory_hash="d" * 64,
    )

    with pytest.raises(RuntimeError, match="second submission failed"):
        submit_authorized_phase(
            dag=dag,
            phase="gate_b",
            preflight_path=preflight,
            authorization_path=authorization,
            submission_record_path=record,
        )

    journal = record.with_suffix(".submission_journal.jsonl")
    rows = [
        json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["step_id"] == "one"
    assert rows[0]["job_id"] == "101"
    assert rows[1]["step_id"] == "two"
    assert rows[1]["returncode"] == 1
    assert not record.exists()
    assert all("PYTHONPATH" not in env for env in scheduler_environments)
    abort = json.loads(
        record.with_suffix(".submission_abort.json").read_text(encoding="utf-8")
    )
    assert abort["status"] == "SUBMISSION_ABORTED"
    assert abort["submitted_job_ids"] == ["101"]
    assert abort["cancellation_returncode"] == 0
