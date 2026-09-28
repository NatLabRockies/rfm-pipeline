"""End-to-end control contracts for the one-shot final G11 campaign."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.g11_final_execution import (
    _development_phase_budget_guard,
    _load_scientific_interaction_runtime,
    _prepare_artifact_job,
    _promote_resolution_cache,
    _resolution_cache_evidence_identity,
    _submit_artifact_job,
    _validate_rfm_runtime_binding,
    advance_campaign,
    inspect_submitted_phase,
    render_final_campaign_config,
)
from scripts.g11_campaign_workflow import _stable_hash


def test_final_controller_direct_cli_entry_point_loads() -> None:
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(repo / "scripts" / "g11_final_execution.py"), "--help"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "{initialize,advance,watch}" in completed.stdout

    advance_help = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "g11_final_execution.py"),
            "advance",
            "--help",
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert advance_help.returncode == 0, advance_help.stderr
    assert "--remaining-au" in advance_help.stdout

    initialize_help = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "g11_final_execution.py"),
            "initialize",
            "--help",
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert initialize_help.returncode == 0, initialize_help.stderr
    assert "--rfm-controller-root" in initialize_help.stdout
    assert "--pilot-resource-freeze" in initialize_help.stdout
    assert "--resolution-cache-evidence-root" in initialize_help.stdout

    runbook = (repo / "docs" / "G11_FINAL_EXECUTION.md").read_text(encoding="utf-8")
    assert "PYTHONPATH=" not in runbook


def test_resolution_cache_evidence_identity_is_content_bound(tmp_path: Path) -> None:
    evidence = tmp_path / "failed-run-evidence"
    evidence.mkdir()
    sums = evidence / "SHA256SUMS"
    sums.write_text("a" * 64 + "  ./failure_status.json\n", encoding="utf-8")
    sums_hash = __import__("hashlib").sha256(sums.read_bytes()).hexdigest()
    (evidence / "SHA256SUMS.sha256").write_text(
        f"{sums_hash}  SHA256SUMS\n", encoding="utf-8"
    )

    identity = _resolution_cache_evidence_identity(evidence)

    assert identity["resolution_cache_evidence_root"] == str(evidence.resolve())
    assert identity["resolution_cache_sha256s_sha256"] == sums_hash
    assert identity["resolution_cache_file_count"] == 1

    sums.write_text("b" * 64 + "  ./failure_status.json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256SUMS"):
        _resolution_cache_evidence_identity(evidence)


def test_completed_resolution_scores_promote_without_scientific_reexecution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib
    import shutil

    import numpy as np
    from rfm_pipeline.campaign_contract import (
        fixed_family_pair_order,
        load_contract,
    )
    import rfm_pipeline.interaction_contract as interaction_contract

    from scripts import g11_campaign_adapter as adapter

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")

    scientific_rfm_root = tmp_path / "scientific-rfm"
    scientific_package = scientific_rfm_root / "src/rfm_pipeline"
    scientific_package.mkdir(parents=True)
    shutil.copy2(
        Path(interaction_contract.__file__).resolve(),
        scientific_package / "interaction_contract.py",
    )
    (scientific_rfm_root / "pixi.lock").write_text(
        "test-only-scientific-lock\n", encoding="utf-8"
    )
    scientific_interaction = _load_scientific_interaction_runtime(scientific_rfm_root)
    scientific_source_hash, scientific_lock_hash = (
        scientific_interaction._current_source_and_lock_hashes()
    )
    monkeypatch.setattr(
        interaction_contract,
        "_current_source_and_lock_hashes",
        lambda: (scientific_source_hash, scientific_lock_hash),
    )

    contract_path = (
        Path(__file__).resolve().parents[1] / "configs/g11_campaign_contract.toml"
    )
    contract, contract_hash = load_contract(contract_path)
    pair_order = fixed_family_pair_order(contract)
    target_contract_dir = tmp_path / "target-package" / "contract"
    target_contract_dir.mkdir(parents=True)
    target_contract = target_contract_dir / "g11_campaign_contract.toml"
    shutil.copy2(contract_path, target_contract)
    family_order = target_contract_dir / "fixed_family_pair_order.json"
    family_order.write_text(json.dumps(list(pair_order)), encoding="utf-8")
    family_order_sha256 = hashlib.sha256(family_order.read_bytes()).hexdigest()
    adapter_path = Path(adapter.__file__).resolve()
    adapter_sha256 = hashlib.sha256(adapter_path.read_bytes()).hexdigest()

    evidence = tmp_path / "evidence"
    source_results = evidence / "scratch_run/stages/resolution/results"
    source_manifest = (
        evidence / "control_root/development_package/stages/resolution/manifest.jsonl"
    )
    source_manifest.parent.mkdir(parents=True)
    runtime_source = evidence / "runtime_source"
    runtime_source.mkdir(parents=True)
    shutil.copy2(adapter_path, runtime_source / "g11_campaign_adapter.py")
    target_results = tmp_path / "target-run/stages/resolution/results"
    target_manifest = tmp_path / "target-package/stages/resolution/manifest.jsonl"
    target_manifest.parent.mkdir(parents=True)

    source_records = []
    target_records = []
    source_hash = None
    lock_hash = None
    for index in range(20):
        shard_id = f"task-{index:04d}"
        seed = 10_000 + index
        spec = adapter._interaction_spec(
            contract,
            draws=contract.resolution_base_draws * contract.resolution_max_multiplier,
            seed=seed,
            n_jobs=4,
        )
        canonical = interaction_contract.canonical_execution_contract_from_specs(spec)
        selected_pairs = pair_order[: contract.resolution_family_size]
        snapshot = interaction_contract.build_control_snapshot(
            canonical,
            candidate_pair_names=selected_pairs,
            training_sample_ids=np.array(["row-0", "row-1"]),
            feature_matrix=np.zeros((2, 2)),
            response_matrix=np.zeros((2, 1)),
            component_names=("component-0",),
        )
        source_hash = snapshot.implementation_source_sha256
        lock_hash = snapshot.dependency_lock_sha256
        draws = contract.resolution_base_draws * contract.resolution_max_multiplier
        artifact = interaction_contract.ScoreOnlyInteractionArtifact(
            status="score_only_completed",
            draw_range_start=0,
            draw_range_end=draws,
            pair_names=selected_pairs,
            observed_scores=np.zeros(len(selected_pairs)),
            null_scores=np.zeros((draws, len(selected_pairs))),
            draw_ids=np.arange(draws),
            control_snapshot=snapshot,
        )
        artifact.write_to(source_results / shard_id / "interaction_score_blocks")
        common = {
            "schema_version": 1,
            "stage": "resolution",
            "shard_id": shard_id,
            "interaction_sharding": "draw-block",
            "status": "pending",
            "attempt": 0,
            "max_retries": 1,
            "contract_config_path": str(target_contract),
            "worker_resources": {
                "estimated_cpu_cores": 1,
                "estimated_memory_gb": 1,
                "estimated_walltime_seconds": 1,
                "requested_cpu_cores": 4,
                "requested_memory_gb": 1,
                "requested_walltime_seconds": 1,
            },
            "expected_range_start": index,
            "expected_range_end": index + 1,
            "operation": "resolution",
            "fixture_kind": ("nondegenerate_null" if index < 10 else "strong_planted"),
            "schedule_index": index % 10,
            "seed": seed,
            "base_draws": contract.resolution_base_draws,
            "nested_schedule_draws": draws,
            "family_size": contract.resolution_family_size,
            "family_order_path": str(family_order),
            "family_order_file_sha256": family_order_sha256,
            "source_hash": source_hash,
            "config_hash": contract_hash,
            "lock_hash": lock_hash,
            "bsm_recovery_driver_sha256": "d" * 64,
            "bsm_dgp_contract_sha256": "e" * 64,
            "scientific_adapter_path": str(adapter_path),
            "scientific_adapter_sha256": adapter_sha256,
            "parent_hash": "f" * 64,
        }
        source_output = tmp_path / "discarded-source" / shard_id
        target_output = target_results / shard_id
        source_records.append(
            {
                **common,
                "output_dir": str(source_output),
                "success_marker": str(source_output / "_SUCCESS.json"),
                "input_hash": hashlib.sha256(f"source-{index}".encode()).hexdigest(),
                "schedule_hash": hashlib.sha256(
                    f"schedule-{index}".encode()
                ).hexdigest(),
                "output_hash": hashlib.sha256(
                    f"old-output-{index}".encode()
                ).hexdigest(),
            }
        )
        target_records.append(
            {
                **common,
                "output_dir": str(target_output),
                "success_marker": str(target_output / "_SUCCESS.json"),
                "input_hash": hashlib.sha256(f"target-{index}".encode()).hexdigest(),
                "schedule_hash": hashlib.sha256(
                    f"new-schedule-{index}".encode()
                ).hexdigest(),
                "output_hash": hashlib.sha256(
                    f"new-output-{index}".encode()
                ).hexdigest(),
            }
        )
    source_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in source_records),
        encoding="utf-8",
    )
    target_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in target_records),
        encoding="utf-8",
    )
    for name, payload in {
        "failure_status.json": {"status": "STOPPED_AFTER_SCIENTIFIC_FAILURE"},
        "failure_diagnosis.json": {
            "status": "CONFIRMED_ROOT_CAUSE",
            "worker_failures_with_identical_signature": 20,
            "canonical_property": "payload_sha256",
        },
        "failure_accounting.json": {
            "status": "REJECTED_SCIENTIFIC_ADAPTER_FAILURE_ACCOUNTED",
            "telemetry_eligibility": "DIAGNOSTIC_ONLY_NEVER_REUSE_FOR_RESOURCE_SELECTION",
        },
    }.items():
        (evidence / name).write_text(json.dumps(payload), encoding="utf-8")
    inventory_rows = []
    for path in sorted(item for item in evidence.rglob("*") if item.is_file()):
        relative = path.relative_to(evidence).as_posix()
        inventory_rows.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{relative}\n"
        )
    sums = evidence / "SHA256SUMS"
    sums.write_text("".join(inventory_rows), encoding="utf-8")
    sums_sha256 = hashlib.sha256(sums.read_bytes()).hexdigest()
    (evidence / "SHA256SUMS.sha256").write_text(
        f"{sums_sha256}  SHA256SUMS\n", encoding="utf-8"
    )

    campaign_root = tmp_path / "control"
    authorization = campaign_root / "development/authorization.json"
    authorization.parent.mkdir(parents=True)
    authorization.write_text("{}\n", encoding="utf-8")
    stage = SimpleNamespace(
        name="resolution",
        job_count=20,
        manifest_path=target_manifest,
        output_root=target_results,
    )
    development = SimpleNamespace(
        stages=(stage,),
        contract_config_path=target_contract,
        config_hash=contract_hash,
        run_id="g11-cache-continuation-test",
    )
    cache_identity = _resolution_cache_evidence_identity(evidence)
    control = {
        "campaign_root": str(campaign_root),
        "rfm_repo_root": str(scientific_rfm_root),
        **cache_identity,
    }
    monkeypatch.setattr(
        interaction_contract,
        "_current_source_and_lock_hashes",
        lambda: ("9" * 64, "8" * 64),
    )

    result = _promote_resolution_cache(
        control=control,
        development=development,
    )

    assert result["status"] == "RESOLUTION_CACHE_PROMOTED"
    assert len(result["promoted_shards"]) == 20
    for record in target_records:
        shard = Path(record["output_dir"])
        worker = json.loads((shard / "result.json").read_text(encoding="utf-8"))
        assert worker["executed_work_units"] == 0
        assert worker["profile_id"] == "verified-cache-promotion"
        assert worker["slurm_job_id"] is None
        assert (
            worker["artifact_checksum"]
            == result["promoted_shards"][int(record["shard_id"].split("-")[-1])][
                "score_payload_sha256"
            ]
        )
        assert (shard / "_SUCCESS.json").is_file()


def test_final_controller_requires_the_pinned_rfm_controller(tmp_path: Path) -> None:
    controller_root = tmp_path / "rfm-controller"
    expected = controller_root / "src" / "rfm_pipeline" / "hpc_campaign_package.py"
    expected.parent.mkdir(parents=True)
    expected.write_text("# frozen runtime marker\n", encoding="utf-8")
    required = {
        "collect_pilot_accounting",
        "generate_campaign_package",
        "select_pilot_resources",
    }

    assert (
        _validate_rfm_runtime_binding(
            controller_root,
            module_path=expected,
            available_names=required,
        )
        == expected.resolve()
    )
    with pytest.raises(RuntimeError, match="pinned RFM controller runtime"):
        _validate_rfm_runtime_binding(
            controller_root,
            module_path=tmp_path / "site-packages" / "hpc_campaign_package.py",
            available_names=required,
        )
    with pytest.raises(RuntimeError, match="collect_pilot_accounting"):
        _validate_rfm_runtime_binding(
            controller_root,
            module_path=expected,
            available_names=required - {"collect_pilot_accounting"},
        )


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def _submission(path: Path) -> None:
    from scripts.g11_campaign_workflow import _stable_hash

    identity = {
        "schema_version": 1,
        "status": "SUBMITTED",
        "phase": "gate_b",
        "run_id": "g11-final-20260815a",
        "source_hash": "a" * 64,
        "config_hash": "b" * 64,
        "lock_hash": "c" * 64,
        "campaign_inventory_hash": "d" * 64,
        "preflight_sha256": "e" * 64,
        "authorization_sha256": "f" * 64,
        "submission_plan_sha256": "1" * 64,
        "submission_job_ids": {"gate_b-worker": "101", "gate_b-reducer": "102"},
    }
    path.write_text(
        json.dumps({**identity, "submission_record_sha256": _stable_hash(identity)}),
        encoding="utf-8",
    )


def test_final_config_gets_unique_collision_safe_run_identity(tmp_path: Path) -> None:
    base = tmp_path / "base.yml"
    base.write_text(
        """hpc_campaign:
  run_id: g11-kestrel-nosubmit
  scheduler_submission_permitted: false
  cluster:
    account: nationalpfa
    scratch_template: /scratch/{user}/bsm_runs/{run_id}
""",
        encoding="utf-8",
    )
    destination = tmp_path / "campaign.yml"

    render_final_campaign_config(
        base_config_path=base,
        output_path=destination,
        run_id="g11-final-20260815a",
    )

    rendered = destination.read_text(encoding="utf-8")
    assert "run_id: g11-final-20260815a" in rendered
    assert "scheduler_submission_permitted: false" in rendered
    assert "scratch_template: /scratch/{user}/bsm_runs/{run_id}" in rendered

    with pytest.raises(ValueError, match="unique final campaign ID"):
        render_final_campaign_config(
            base_config_path=base,
            output_path=tmp_path / "bad.yml",
            run_id="g11-kestrel-nosubmit",
        )
    with pytest.raises(ValueError, match="refusing to overwrite"):
        render_final_campaign_config(
            base_config_path=base,
            output_path=destination,
            run_id="g11-final-20260815b",
        )


def test_phase_probe_waits_until_jobs_leave_queue_and_sacct_is_complete(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "submission.json"
    _submission(submission)
    calls = iter(
        [
            _completed("101|RUNNING\n102|PENDING\n"),
            _completed(""),
            _completed("101|COMPLETED|0:0\n"),
        ]
    )

    waiting = inspect_submitted_phase(
        submission, run_command=lambda *_a, **_k: next(calls)
    )
    assert waiting["status"] == "WAITING_FOR_SCHEDULER"

    accounting_lag = inspect_submitted_phase(
        submission, run_command=lambda *_a, **_k: next(calls)
    )
    assert accounting_lag["status"] == "WAITING_FOR_SACCT"


def test_phase_probe_fails_closed_on_any_terminal_scheduler_failure(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "submission.json"
    _submission(submission)
    calls = iter(
        [
            _completed(""),
            _completed("101|COMPLETED|0:0\n102|OUT_OF_MEMORY|0:125\n"),
        ]
    )

    with pytest.raises(RuntimeError, match="OUT_OF_MEMORY/0:125"):
        inspect_submitted_phase(submission, run_command=lambda *_a, **_k: next(calls))


def test_phase_probe_marks_only_exact_completion_ready_for_verification(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "submission.json"
    _submission(submission)
    calls = iter(
        [
            _completed(""),
            _completed(
                "101|COMPLETED|0:0\n101.batch|COMPLETED|0:0\n102|COMPLETED|0:0\n"
            ),
        ]
    )

    observed = inspect_submitted_phase(
        submission, run_command=lambda *_a, **_k: next(calls)
    )
    assert observed["status"] == "READY_TO_VERIFY"
    assert observed["job_count"] == 2


def test_phase_probe_accepts_exact_array_tasks_without_a_parent_sacct_row(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "submission.json"
    _submission(submission)
    calls = iter(
        [
            _completed(""),
            _completed(
                "101_0|COMPLETED|0:0\n"
                "101_0.batch|COMPLETED|0:0\n"
                "101_1|COMPLETED|0:0\n"
                "102|COMPLETED|0:0\n"
            ),
        ]
    )
    commands: list[list[str]] = []

    def run_command(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return next(calls)

    observed = inspect_submitted_phase(
        submission,
        expected_array_task_counts={"gate_b-worker": 2, "gate_b-reducer": 0},
        run_command=run_command,
    )

    assert observed["status"] == "READY_TO_VERIFY"
    assert observed["job_count"] == 2
    assert "--array" in commands[1]
    assert "--format=JobID,State,ExitCode" in commands[1]


def test_advance_uses_fresh_per_invocation_remaining_au(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "control"
    (root / "development_package").mkdir(parents=True)
    (root / "development_package" / "package_summary.json").write_text(
        "{}", encoding="utf-8"
    )
    (root / "resource_freeze.json").write_text("{}", encoding="utf-8")
    config = root / "final_campaign.yml"
    config.write_text("config", encoding="utf-8")
    repo = tmp_path / "rfm"
    repo.mkdir()
    control = {
        "campaign_root": str(root),
        "final_config_path": str(config),
        "rfm_repo_root": str(repo),
        "remaining_au_for_live_preflight": 25_000,
    }
    development = SimpleNamespace(run_id="g11-final")
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "scripts.g11_final_execution._load_control", lambda _path: control
    )
    monkeypatch.setattr(
        "scripts.g11_final_execution.load_packaged_dag", lambda **_kwargs: development
    )

    def fake_advance_phase(**kwargs: object) -> dict[str, str]:
        observed.update(kwargs)
        return {"status": "EXECUTION_FLAG_REQUIRED", "phase": "development"}

    monkeypatch.setattr(
        "scripts.g11_final_execution._advance_phase", fake_advance_phase
    )
    monkeypatch.setattr(
        "scripts.g11_final_execution._development_phase_budget_guard",
        lambda **_kwargs: {
            "status": "DEVELOPMENT_WITHIN_ALLOCATION",
            "development_admission_sha256": "a" * 64,
        },
    )

    result = advance_campaign(
        tmp_path / "control_manifest.json",
        execute=True,
        remaining_au=12_345,
    )

    assert result == {
        "status": "EXECUTION_FLAG_REQUIRED",
        "phase": "development",
    }
    assert observed["remaining_au"] == 12_345

    with pytest.raises(ValueError, match="live remaining AU must be positive"):
        advance_campaign(
            tmp_path / "control_manifest.json",
            execute=True,
            remaining_au=0,
        )


def test_development_phase_has_a_whole_campaign_admission_guard(tmp_path: Path) -> None:
    freeze_identity = {
        "allocation_accounting": {
            "accepted_pilot_observed_au": 90.36944444444445,
            "prior_rejected_attempt_au": 192.55,
        }
    }
    freeze = {
        **freeze_identity,
        "resource_freeze_sha256": _stable_hash(freeze_identity),
    }
    freeze_path = tmp_path / "resource_freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    dag = SimpleNamespace(
        campaign_envelope=SimpleNamespace(
            stage_allocations=(
                SimpleNamespace(stage_name="resolution", requested_au=300.0),
            )
        )
    )

    guard = _development_phase_budget_guard(
        dag=dag,
        resource_freeze_path=freeze_path,
        control={"postprocessing_reserved_au": 5.0, "allocation_quota_au": 25_000.0},
    )

    assert guard["status"] == "DEVELOPMENT_WITHIN_ALLOCATION"
    assert guard["development_requested_au"] == 300.0


def test_controller_never_crosses_a_phase_or_publication_gate_out_of_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "control"
    root.mkdir()
    config = root / "final_campaign.yml"
    config.write_text("config", encoding="utf-8")
    rfm = tmp_path / "rfm"
    bsm = tmp_path / "bsm"
    pilot_package = tmp_path / "pilot-package"
    pilot_submission = tmp_path / "pilot-submission.json"
    pilot_resource_freeze = tmp_path / "pilot-resource-freeze.json"
    for directory in (rfm, bsm, pilot_package):
        directory.mkdir()
    pilot_submission.write_text("{}", encoding="utf-8")
    pilot_resource_freeze.write_text("{}", encoding="utf-8")
    control = {
        "campaign_root": str(root),
        "final_config_path": str(config),
        "rfm_repo_root": str(rfm),
        "bsm_runtime_root": str(bsm),
        "pilot_package_root": str(pilot_package),
        "pilot_submission_record": str(pilot_submission),
        "pilot_resource_freeze": str(pilot_resource_freeze),
        "prior_sunk_au": 114.63055555555556,
        "remaining_au_for_live_preflight": 25_000,
        "allocation_quota_au": 25_000.0,
        "postprocessing_reserved_au": 5.0,
    }
    dags = {
        str(pilot_package): SimpleNamespace(name="pilot"),
        str(root / "development_package"): SimpleNamespace(name="development"),
        str(root / "final_confirmatory_package"): SimpleNamespace(name="final"),
    }
    phase_state = {
        "development": "PHASE_COMPLETE",
        "gate_b": "WAITING_FOR_SCHEDULER",
        "gate_p": "WAITING_FOR_SCHEDULER",
        "gate_c": "WAITING_FOR_SCHEDULER",
    }
    phase_calls: list[str] = []

    monkeypatch.setattr(
        "scripts.g11_final_execution._load_control", lambda _path: control
    )
    monkeypatch.setattr(
        "scripts.g11_final_execution.load_packaged_dag",
        lambda **kwargs: dags[str(kwargs["package_root"])],
    )

    def fake_close(**_kwargs: object) -> None:
        (root / "resource_freeze.json").write_text("{}", encoding="utf-8")

    def fake_prepare_development(**_kwargs: object) -> None:
        package = root / "development_package"
        package.mkdir()
        (package / "package_summary.json").write_text("{}", encoding="utf-8")

    def fake_prepare_final(**_kwargs: object) -> None:
        package = root / "final_confirmatory_package"
        package.mkdir()
        (package / "package_summary.json").write_text("{}", encoding="utf-8")
        (root / "budget_certificate.json").write_text("{}", encoding="utf-8")

    def fake_advance_phase(**kwargs: object) -> dict[str, str]:
        phase = str(kwargs["phase"])
        phase_calls.append(phase)
        return {"status": phase_state[phase], "phase": phase}

    def fake_prepare_artifact(**_kwargs: object) -> None:
        job = root / "publication_job"
        job.mkdir()
        (job / "artifact_job_inputs.json").write_text("{}", encoding="utf-8")

    def fake_submit(_root: Path) -> None:
        (root / "publication_job" / "submission.json").write_text(
            "{}", encoding="utf-8"
        )

    def fake_verify(_root: Path) -> dict[str, str]:
        (root / "publication_job" / "completion.json").write_text(
            "{}", encoding="utf-8"
        )
        return {"status": "HPC_AND_PUBLICATION_BUNDLE_COMPLETE"}

    monkeypatch.setattr("scripts.g11_final_execution.close_completed_pilot", fake_close)
    monkeypatch.setattr(
        "scripts.g11_final_execution.prepare_development_package",
        fake_prepare_development,
    )
    monkeypatch.setattr(
        "scripts.g11_final_execution.prepare_final_package", fake_prepare_final
    )
    monkeypatch.setattr(
        "scripts.g11_final_execution._advance_phase", fake_advance_phase
    )
    monkeypatch.setattr(
        "scripts.g11_final_execution._stage_artifact",
        lambda _dag, stage, filename: root / stage / filename,
    )
    monkeypatch.setattr(
        "scripts.g11_final_execution._validate_budget_certificate",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        "scripts.g11_final_execution._confirmatory_phase_budget_guard",
        lambda **_kwargs: {
            "status": "PHASE_WITHIN_ALLOCATION",
            "phase_budget_guard_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        "scripts.g11_final_execution._development_phase_budget_guard",
        lambda **_kwargs: {
            "status": "DEVELOPMENT_WITHIN_ALLOCATION",
            "development_admission_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(
        "scripts.g11_final_execution._prepare_artifact_job", fake_prepare_artifact
    )
    monkeypatch.setattr("scripts.g11_final_execution._submit_artifact_job", fake_submit)
    monkeypatch.setattr(
        "scripts.g11_final_execution.inspect_submitted_phase",
        lambda _path: {"status": "READY_TO_VERIFY"},
    )
    monkeypatch.setattr(
        "scripts.g11_final_execution._verify_publication_job", fake_verify
    )
    monkeypatch.setattr(
        "scripts.g11_final_execution._validate_publication_completion",
        lambda _root: {"status": "HPC_AND_PUBLICATION_BUNDLE_COMPLETE"},
    )

    manifest = tmp_path / "control_manifest.json"
    assert advance_campaign(manifest, execute=False)["status"] == (
        "PILOT_CLOSED_AND_RESOURCES_FROZEN"
    )
    assert advance_campaign(manifest, execute=False)["status"] == (
        "DEVELOPMENT_PACKAGE_READY"
    )
    assert advance_campaign(manifest, execute=False)["status"] == (
        "FINAL_CONFIRMATORY_PACKAGE_BUDGET_CERTIFIED"
    )
    assert phase_calls == ["development"]

    assert advance_campaign(manifest, execute=True)["phase"] == "gate_b"
    assert phase_calls[-1:] == ["gate_b"]
    phase_state["gate_b"] = "PHASE_COMPLETE"
    assert advance_campaign(manifest, execute=True)["phase"] == "gate_p"
    assert phase_calls[-2:] == ["gate_b", "gate_p"]
    phase_state["gate_p"] = "PHASE_COMPLETE"
    assert advance_campaign(manifest, execute=True)["phase"] == "gate_c"
    assert phase_calls[-3:] == ["gate_b", "gate_p", "gate_c"]
    phase_state["gate_c"] = "PHASE_COMPLETE"

    assert advance_campaign(manifest, execute=True)["status"] == (
        "PUBLICATION_JOB_PREPARED_NO_SUBMIT"
    )
    assert advance_campaign(manifest, execute=False)["status"] == (
        "EXECUTION_FLAG_REQUIRED"
    )
    assert advance_campaign(manifest, execute=True)["status"] == (
        "PUBLICATION_JOB_SUBMITTED"
    )
    assert advance_campaign(manifest, execute=True)["status"] == (
        "HPC_AND_PUBLICATION_BUNDLE_COMPLETE"
    )
    assert advance_campaign(manifest, execute=True)["status"] == (
        "HPC_AND_PUBLICATION_BUNDLE_COMPLETE"
    )


def _artifact_job_fixture(tmp_path: Path) -> tuple[Path, SimpleNamespace, Path]:
    campaign_root = tmp_path / "campaign"
    package_root = tmp_path / "package"
    results_root = tmp_path / "results"
    reducer = results_root / "stages" / "gate_b" / "reducer"
    reducer.mkdir(parents=True)
    package_root.mkdir()
    bsm_runtime = tmp_path / "bsm-runtime"
    scientific_runtime = tmp_path / "rfm-scientific"
    python = scientific_runtime / ".pixi" / "envs" / "default" / "bin" / "python"
    builder = bsm_runtime / "scripts" / "build_g11_publication_artifacts.py"
    python.parent.mkdir(parents=True)
    builder.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    builder.write_text("print('builder')\n", encoding="utf-8")
    dag = SimpleNamespace(
        run_id="g11-final-20260815a",
        output_dir=package_root,
        repo_root=scientific_runtime,
        campaign_inventory_hash="a" * 64,
        stages=(SimpleNamespace(reducer_output_dir=reducer),),
    )
    return campaign_root, dag, bsm_runtime


def test_publication_job_script_has_exactly_one_value_for_each_cli_option(
    tmp_path: Path,
) -> None:
    root, dag, bsm_runtime = _artifact_job_fixture(tmp_path)

    inputs = _prepare_artifact_job(
        root=root,
        dag=dag,
        bsm_runtime_root=bsm_runtime,
    )

    script = Path(inputs["script_path"]).read_text(encoding="utf-8")
    command = shlex.split(script.splitlines()[-1])
    assert command.count("--package-root") == 1
    assert command.count("--results-root") == 1
    assert command.count("--output-root") == 1
    assert command[0] == str(dag.repo_root / ".pixi/envs/default/bin/python")
    assert command[command.index("--results-root") + 1] == str(tmp_path / "results")


def test_publication_submission_journals_returned_job_id_before_final_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.g11_campaign_workflow import _sha256_path

    root, dag, bsm_runtime = _artifact_job_fixture(tmp_path)
    _prepare_artifact_job(root=root, dag=dag, bsm_runtime_root=bsm_runtime)
    monkeypatch.setenv("PYTHONPATH", "/tmp/wrong-rfm")
    observed: dict[str, object] = {}
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        observed.update(kwargs)
        commands.append(command)
        if "--test-only" in command:
            return subprocess.CompletedProcess(
                [], 0, stdout="test accepted\n", stderr=""
            )
        return subprocess.CompletedProcess([], 0, stdout="12345;cluster\n", stderr="")

    monkeypatch.setattr(
        "scripts.g11_final_execution.subprocess.run",
        fake_run,
    )

    submission = _submit_artifact_job(root)

    assert commands[0][0:2] == ["sbatch", "--test-only"]
    assert commands[1][0:2] == ["sbatch", "--parsable"]
    preflight = json.loads(
        (root / "publication_job" / "submission_preflight.json").read_text(
            encoding="utf-8"
        )
    )
    assert preflight["status"] == "SBATCH_TEST_ONLY_PASSED"
    journal = root / "publication_job" / "submission_journal.jsonl"
    assert journal.is_file()
    with journal.open(encoding="utf-8") as handle:
        os.fsync(handle.fileno())
        entry = json.loads(handle.read())
    assert entry["job_id"] == "12345"
    assert entry["returncode"] == 0
    assert submission["submission_journal_sha256"] == _sha256_path(journal)
    assert (
        submission["submission_preflight_sha256"]
        == preflight["submission_preflight_sha256"]
    )
    assert "PYTHONPATH" not in observed["env"]


def test_publication_submission_stops_when_sbatch_test_only_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, dag, bsm_runtime = _artifact_job_fixture(tmp_path)
    _prepare_artifact_job(root=root, dag=dag, bsm_runtime_root=bsm_runtime)
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess([], 1, stdout="", stderr="invalid account")

    monkeypatch.setattr("scripts.g11_final_execution.subprocess.run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        _submit_artifact_job(root)

    assert len(calls) == 1
    assert calls[0][1] == "--test-only"
    assert not (root / "publication_job" / "submission_journal.jsonl").exists()
    preflight = json.loads(
        (root / "publication_job" / "submission_preflight.json").read_text(
            encoding="utf-8"
        )
    )
    assert preflight["status"] == "SBATCH_TEST_ONLY_FAILED"
