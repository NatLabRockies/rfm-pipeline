"""Fail-closed orchestration helpers for the final G11 Kestrel campaign.

This module deliberately lives outside ``rfm_pipeline`` so that improving the
controller cannot change the source hash measured by the active pilot.  It
consumes immutable package/evidence bytes and delegates all scientific work to
the content-addressed RFM/BSM campaign implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

PUBLICATION_FIXED_FAMILY_REPLICATES = 300
SUPERSEDED_FIXED_FAMILY_CONTRACT_SHA256 = (
    "66f9a7fb702c0726c464393f7c153001846510d891a75ccb2c1d48b467375c24"
)
PUBLICATION_B999_CONFIRMATORY_ESTIMATED_AU = 20_250.751923076525
PUBLICATION_B999_CONFIRMATORY_REQUESTED_AU = math.ceil(
    PUBLICATION_B999_CONFIRMATORY_ESTIMATED_AU * 1.20
)


def _scheduler_environment() -> dict[str, str]:
    """Return an import-clean environment for every real Slurm submission."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _sacct_array_command(
    submission_job_ids: dict[str, str], *, fields: tuple[str, ...]
) -> list[str]:
    """Request canonical array identities from Slurm accounting."""
    if not submission_job_ids or not fields:
        raise ValueError("sacct command requires job IDs and fields")
    job_ids = [str(value) for value in submission_job_ids.values()]
    if any(re.fullmatch(r"[0-9]+", value) is None for value in job_ids):
        raise ValueError("sacct command received a malformed scheduler job ID")
    return [
        "sacct",
        "--array",
        "-j",
        ",".join(sorted(job_ids, key=int)),
        "-nP",
        f"--format={','.join(fields)}",
    ]


def _stable_hash(payload: Any) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_completed_submission_record(
    path: str | Path,
    *,
    expected_step_ids: set[str],
    expected_identity: dict[str, str],
) -> dict[str, str]:
    """Return exact scheduler IDs only from one terminal, identity-bound record."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE":
        raise ValueError("submission record is not COMPLETE")
    for field, expected in expected_identity.items():
        if payload.get(field) != expected:
            raise ValueError(f"submission record {field} differs from the package")
    job_ids = payload.get("submission_job_ids")
    if not isinstance(job_ids, dict) or set(job_ids) != expected_step_ids:
        raise ValueError("submission job IDs do not exactly cover the packaged plan")
    normalized = {str(key): str(value) for key, value in job_ids.items()}
    if any(re.fullmatch(r"[0-9]+", value) is None for value in normalized.values()):
        raise ValueError("submission record contains a malformed scheduler job ID")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("submission record reuses a scheduler job ID")
    return normalized


def validate_scheduler_completion(
    submission_job_ids: dict[str, str], raw_sacct: str
) -> dict[str, dict[str, str]]:
    """Require one exact top-level COMPLETED/0:0 row for every submitted step."""
    by_job_id = {
        str(job_id): str(step_id) for step_id, job_id in submission_job_ids.items()
    }
    if len(by_job_id) != len(submission_job_ids):
        raise ValueError("phase submission reuses a scheduler job ID")
    top_level: dict[str, tuple[str, str]] = {}
    for line in raw_sacct.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        if len(values) != 3:
            raise ValueError("phase sacct output does not match JobID/State/ExitCode")
        job_id, state, exit_code = values
        if job_id not in by_job_id:
            continue
        if job_id in top_level:
            raise ValueError(f"phase sacct output duplicates top-level job {job_id}")
        top_level[job_id] = (state, exit_code)
    if set(top_level) != set(by_job_id):
        raise ValueError(
            "phase sacct output lacks exact top-level submitted-job coverage"
        )
    observed: dict[str, dict[str, str]] = {}
    for job_id, step_id in by_job_id.items():
        state, exit_code = top_level[job_id]
        if state != "COMPLETED" or exit_code != "0:0":
            raise ValueError(
                f"phase step {step_id} is not exactly COMPLETED/0:0: "
                f"{state}/{exit_code}"
            )
        observed[step_id] = {
            "job_id": job_id,
            "state": state,
            "exit_code": exit_code,
        }
    return observed


def validate_scheduler_completion_accounting(
    submission_job_ids: dict[str, str],
    raw_sacct: str,
    *,
    cpu_charge_factor: float,
    qos_factor: float,
    step_accounting: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], float]:
    """Validate terminal jobs and sum every billed allocation, including arrays."""
    if cpu_charge_factor <= 0 or qos_factor <= 0:
        raise ValueError("scheduler accounting charge factors must be positive")
    by_job_id = {
        str(job_id): str(step_id) for step_id, job_id in submission_job_ids.items()
    }
    if len(by_job_id) != len(submission_job_ids):
        raise ValueError("phase submission reuses a scheduler job ID")
    rows: dict[str, tuple[str, str, int, int]] = {}
    for line in raw_sacct.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        if len(values) != 5:
            raise ValueError(
                "phase sacct accounting output does not match "
                "JobID/State/ExitCode/ElapsedRaw/AllocNodes"
            )
        job_id, state, exit_code, raw_elapsed, raw_nodes = values
        relevant = job_id in by_job_id or any(
            re.fullmatch(rf"{re.escape(parent)}_[0-9]+", job_id) for parent in by_job_id
        )
        if not relevant:
            continue
        if job_id in rows:
            raise ValueError(
                f"phase sacct accounting duplicates allocation job {job_id}"
            )
        try:
            elapsed = int(raw_elapsed)
            nodes = int(raw_nodes)
        except ValueError as exc:
            raise ValueError(
                f"phase accounting row is nonnumeric for job {job_id}"
            ) from exc
        if elapsed < 0 or nodes < 0:
            raise ValueError(f"phase accounting row is invalid for job {job_id}")
        rows[job_id] = (state, exit_code, elapsed, nodes)
    normalized_accounting = (
        {
            step_id: {"array_task_count": 0, "shared_node_equivalent": None}
            for step_id in submission_job_ids
        }
        if step_accounting is None
        else step_accounting
    )
    if set(normalized_accounting) != set(submission_job_ids):
        raise ValueError(
            "phase accounting metadata lacks exact submitted-step coverage"
        )
    observed: dict[str, dict[str, Any]] = {}
    total_au = 0.0
    for job_id, step_id in by_job_id.items():
        metadata = normalized_accounting[step_id]
        array_task_count = int(metadata.get("array_task_count", 0))
        shared_node_equivalent = metadata.get("shared_node_equivalent")
        if array_task_count < 0:
            raise ValueError("phase accounting array task count is invalid")
        if shared_node_equivalent is not None:
            shared_node_equivalent = float(shared_node_equivalent)
            if not math.isfinite(shared_node_equivalent) or not (
                0 < shared_node_equivalent <= 1
            ):
                raise ValueError("phase shared-node accounting fraction is invalid")
        if array_task_count:
            parent_row = rows.get(job_id)
            if parent_row is not None:
                state, exit_code, _, _ = parent_row
                if state != "COMPLETED" or exit_code != "0:0":
                    raise ValueError(
                        f"phase step {step_id} is not exactly COMPLETED/0:0: "
                        f"{state}/{exit_code}"
                    )
            else:
                # Slurm installations may omit the synthetic array-parent row.
                # Exact terminal acceptance and charging are therefore based on
                # the complete, explicitly enumerated set of array-task rows.
                state, exit_code = "COMPLETED", "0:0"
            expected_task_ids = {
                f"{job_id}_{index}" for index in range(array_task_count)
            }
            observed_task_ids = {
                row_id
                for row_id in rows
                if re.fullmatch(rf"{re.escape(job_id)}_[0-9]+", row_id)
            }
            if observed_task_ids != expected_task_ids:
                raise ValueError(
                    f"phase array-task coverage differs for step {step_id}"
                )
            allocation_rows = [rows[row_id] for row_id in sorted(expected_task_ids)]
        else:
            if job_id not in rows:
                raise ValueError(
                    "phase sacct accounting lacks exact top-level "
                    "submitted-job coverage"
                )
            state, exit_code, parent_elapsed, parent_nodes = rows[job_id]
            if state != "COMPLETED" or exit_code != "0:0":
                raise ValueError(
                    f"phase step {step_id} is not exactly COMPLETED/0:0: "
                    f"{state}/{exit_code}"
                )
            allocation_rows = [(state, exit_code, parent_elapsed, parent_nodes)]
        elapsed = 0
        nodes = 0
        observed_au = 0.0
        for task_state, task_exit, task_elapsed, task_nodes in allocation_rows:
            if task_state != "COMPLETED" or task_exit != "0:0" or task_nodes <= 0:
                raise ValueError(
                    f"phase step {step_id} has an invalid allocation row: "
                    f"{task_state}/{task_exit}/{task_nodes} nodes"
                )
            elapsed += task_elapsed
            nodes += task_nodes
            charged_nodes = (
                shared_node_equivalent
                if shared_node_equivalent is not None
                else float(task_nodes)
            )
            observed_au += (
                task_elapsed / 3600.0 * charged_nodes * cpu_charge_factor * qos_factor
            )
        observed[step_id] = {
            "job_id": job_id,
            "state": state,
            "exit_code": exit_code,
            "elapsed_seconds": elapsed,
            "allocated_nodes": nodes,
            "accounted_task_count": len(allocation_rows),
            "shared_node_equivalent": shared_node_equivalent,
            "observed_au": observed_au,
        }
        total_au += observed_au
    return observed, total_au


def augment_resource_freeze_accounting(
    resource_freeze: dict[str, Any],
    *,
    pilot_accounting: dict[str, Any],
    prior_sunk_au: float,
) -> dict[str, Any]:
    """Bind accepted-pilot and rejected-attempt allocation to the resource freeze."""
    if prior_sunk_au < 0:
        raise ValueError("prior sunk AUs must be nonnegative")
    if pilot_accounting.get("status") != "COMPLETE":
        raise ValueError("pilot accounting is not COMPLETE")
    accounting_hash = str(pilot_accounting.get("pilot_accounting_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", accounting_hash) is None:
        raise ValueError("pilot accounting hash is malformed")
    observed = float(pilot_accounting.get("observed_total_au", -1.0))
    if observed < 0:
        raise ValueError("accepted pilot observed AUs must be nonnegative")
    augmented = deepcopy(resource_freeze)
    augmented.pop("resource_freeze_sha256", None)
    augmented["allocation_accounting"] = {
        "pilot_accounting_sha256": accounting_hash,
        "accepted_pilot_observed_au": observed,
        "prior_rejected_attempt_au": float(prior_sunk_au),
        "spent_through_pilot_au": observed + float(prior_sunk_au),
        "accounting_rule": (
            "accepted pilot observed sacct AUs plus all rejected-attempt AUs; "
            "completed phases use exact sacct AUs, the shared reserve applies only "
            "to unexecuted work, and each next phase must fit at requested walltime"
        ),
    }
    augmented["resource_freeze_sha256"] = _stable_hash(augmented)
    return augmented


def _reconcile_accepted_resource_freeze(
    observed: dict[str, Any], accepted: dict[str, Any]
) -> dict[str, Any]:
    """Preserve accepted pilot science while advancing cumulative sunk AUs.

    The resource selections and allocation values are scientific evidence.  A
    later controller may add post-pilot rejected-attempt charges and improve
    the human-readable accounting-rule text, but it must not replace any pilot
    identity, telemetry, resource selection, or observed pilot charge.
    """

    for label, payload in (("recomputed", observed), ("accepted", accepted)):
        identity = {
            key: value
            for key, value in payload.items()
            if key != "resource_freeze_sha256"
        }
        if payload.get("resource_freeze_sha256") != _stable_hash(identity):
            raise ValueError(f"{label} pilot resource freeze self-hash differs")
        allocation = payload.get("allocation_accounting")
        if not isinstance(allocation, dict) or not isinstance(
            allocation.get("accounting_rule"), str
        ):
            raise TypeError(f"{label} pilot resource freeze lacks an accounting rule")

    observed_science = {
        key: value
        for key, value in observed.items()
        if key not in {"resource_freeze_sha256", "allocation_accounting"}
    }
    accepted_science = {
        key: value
        for key, value in accepted.items()
        if key not in {"resource_freeze_sha256", "allocation_accounting"}
    }
    if observed_science != accepted_science:
        raise ValueError(
            "recomputed pilot resource freeze differs from accepted evidence"
        )

    observed_accounting = observed["allocation_accounting"]
    accepted_accounting = accepted["allocation_accounting"]
    mutable_fields = {
        "prior_rejected_attempt_au",
        "spent_through_pilot_au",
        "accounting_rule",
    }
    if set(observed_accounting) != set(accepted_accounting) or any(
        observed_accounting[field] != accepted_accounting[field]
        for field in set(accepted_accounting) - mutable_fields
    ):
        raise ValueError(
            "recomputed pilot resource freeze differs from accepted evidence"
        )
    accepted_pilot_au = float(accepted_accounting["accepted_pilot_observed_au"])
    observed_pilot_au = float(observed_accounting["accepted_pilot_observed_au"])
    accepted_rejected_au = float(accepted_accounting["prior_rejected_attempt_au"])
    observed_rejected_au = float(observed_accounting["prior_rejected_attempt_au"])
    values = (
        accepted_pilot_au,
        observed_pilot_au,
        accepted_rejected_au,
        observed_rejected_au,
    )
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("pilot resource-freeze allocation values are invalid")
    if observed_pilot_au != accepted_pilot_au:
        raise ValueError(
            "recomputed pilot resource freeze differs from accepted evidence"
        )
    if observed_rejected_au < accepted_rejected_au:
        raise ValueError("cumulative rejected-attempt AUs cannot decrease")
    for label, accounting, rejected_au in (
        ("recomputed", observed_accounting, observed_rejected_au),
        ("accepted", accepted_accounting, accepted_rejected_au),
    ):
        spent = float(accounting["spent_through_pilot_au"])
        if not math.isfinite(spent) or not math.isclose(
            spent, accepted_pilot_au + rejected_au, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{label} pilot cumulative allocation differs")

    normalized = deepcopy(observed)
    normalized["allocation_accounting"]["accounting_rule"] = accepted_accounting[
        "accounting_rule"
    ]
    normalized_identity = {
        key: value
        for key, value in normalized.items()
        if key != "resource_freeze_sha256"
    }
    normalized["resource_freeze_sha256"] = _stable_hash(normalized_identity)
    return normalized


def build_publication_contract_amendment(
    base_contract: Any,
) -> tuple[Any, dict[str, Any]]:
    """Freeze the sole post-pilot scientific amendment permitted for publication.

    The accepted pilot and development resolution retain their original
    1,000-fixed-family base-contract identity.  Before any confirmatory work,
    the non-vacuity supplement is prospectively reduced to 300 complete
    replicates.  The five primary null regimes remain at 1,000 replicates each.
    """
    from rfm_pipeline.campaign_contract import (
        compute_contract_hash,
        largest_passing_wilson_count,
    )

    if int(base_contract.fixed_family_replicates) != 1000:
        raise ValueError(
            "publication amendment requires the accepted 1,000-replicate pilot base"
        )
    if (
        base_contract.schema_version != "g11_campaign_contract_v11"
        or int(base_contract.generation) != 12
        or not math.isclose(
            float(base_contract.tree_family_alpha), 0.020, rel_tol=0.0, abs_tol=1e-12
        )
        or not math.isclose(
            float(base_contract.binary_binary_family_alpha),
            0.020,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "publication amendment requires the prospective generation-12 "
            "0.020/0.020 detector-family contract"
        )
    null_scenarios = tuple(
        scenario for scenario in base_contract.scenarios if scenario.kind == "null"
    )
    if len(null_scenarios) != 5 or any(
        int(scenario.n_replicates) != 1000 for scenario in null_scenarios
    ):
        raise ValueError(
            "publication amendment cannot alter the five 1,000-replicate null regimes"
        )
    publication_contract = replace(
        base_contract,
        fixed_family_replicates=PUBLICATION_FIXED_FAMILY_REPLICATES,
    )
    before = asdict(base_contract)
    after = asdict(publication_contract)
    changed = {key for key in before if before[key] != after[key]}
    if changed != {"fixed_family_replicates"}:
        raise ValueError(
            "publication amendment changed fields outside the fixed-family count"
        )
    identity = {
        "schema_version": 1,
        "status": "PRESPECIFIED_BEFORE_CONFIRMATORY_EXECUTION",
        "effective_phase": "post_resolution_pre_confirmatory",
        "pilot_base_contract_hash": compute_contract_hash(base_contract),
        "publication_base_contract_hash": compute_contract_hash(publication_contract),
        "changed_fields": {
            "fixed_family_replicates": {
                "before": int(base_contract.fixed_family_replicates),
                "after": PUBLICATION_FIXED_FAMILY_REPLICATES,
            }
        },
        "scientific_generation_transition": {
            "superseded_schema_version": "g11_campaign_contract_v10",
            "superseded_generation": 11,
            "superseded_contract_hash": SUPERSEDED_FIXED_FAMILY_CONTRACT_SHA256,
            "selected_schema_version": publication_contract.schema_version,
            "selected_generation": int(publication_contract.generation),
            "changed_scientific_fields": {
                "tree_family_alpha": {"before": 0.025, "after": 0.020},
                "binary_binary_family_alpha": {
                    "before": 0.025,
                    "after": 0.020,
                },
                "fixed_family_replicates": {
                    "before": 200,
                    "after": PUBLICATION_FIXED_FAMILY_REPLICATES,
                },
            },
            "superseded_gate_b_evidence_status": "development_only_not_adoptable",
            "superseded_fixed_family_evidence_status": (
                "failed_confirmatory_not_reusable"
            ),
            "required_fresh_phases": ["gate_b", "fixed_family_supplement"],
            "seed_policy": "contract_hash_derived_zero_overlap_required",
        },
        "primary_null_regimes": {
            "count": len(null_scenarios),
            "replicates_per_regime": 1000,
            "unchanged": True,
        },
        "fixed_family_gate": {
            "family_sizes": list(publication_contract.fixed_family_sizes),
            "replicates": PUBLICATION_FIXED_FAMILY_REPLICATES,
            "confidence": float(publication_contract.calibration_confidence),
            "one_sided_wilson_upper_limit": float(publication_contract.gate_value),
            "largest_passing_false_selection_count": largest_passing_wilson_count(
                n_replicates=PUBLICATION_FIXED_FAMILY_REPLICATES,
                gate_value=float(publication_contract.gate_value),
                confidence=float(publication_contract.calibration_confidence),
            ),
        },
        "execution_rule": (
            "exactly 300 precommitted replicates; no interim testing, optional stopping, "
            "or later incremental expansion"
        ),
        "rationale": (
            "retain the fixed-family non-vacuity check with a prespecified one-sided "
            "confidence-bound decision while keeping the complete manuscript campaign "
            "within the approved campaign allocation"
        ),
    }
    return publication_contract, {
        **identity,
        "contract_amendment_sha256": _stable_hash(identity),
    }


def adopt_generation_12_prerequisites(
    *,
    source_resource_freeze: dict[str, Any],
    source_resolution_decision: dict[str, Any],
    source_resolution_decision_sha256: str,
    source_development_completion: dict[str, Any],
    target_source_hash: str,
    target_config_hash: str,
    target_lock_hash: str,
    target_run_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Adopt only invariant pilot telemetry and the accepted resolution into G12.

    Gate-B and fixed-family results are intentionally not accepted here.  The
    returned records make the unchanged resource choice and B=999 resolution
    explicit while rebinding their controller identities to the prospective
    generation-12 package.
    """
    source_freeze_hash = str(source_resource_freeze.get("resource_freeze_sha256", ""))
    source_freeze_identity = {
        key: value
        for key, value in source_resource_freeze.items()
        if key != "resource_freeze_sha256"
    }
    if (
        re.fullmatch(r"[0-9a-f]{64}", source_freeze_hash) is None
        or _stable_hash(source_freeze_identity) != source_freeze_hash
    ):
        raise ValueError("source resource freeze self-hash differs")

    target_hashes = (target_source_hash, target_config_hash, target_lock_hash)
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in target_hashes):
        raise ValueError("target source, contract, and lock hashes must be SHA-256")
    if not target_run_id.strip():
        raise ValueError("target run identity is empty")
    if target_config_hash == source_resource_freeze.get("config_hash"):
        raise ValueError("generation-12 adoption requires a new contract identity")

    if (
        source_resolution_decision.get("operation") != "resolution"
        or source_resolution_decision.get("status") != "completed"
        or source_resolution_decision.get("decision") != "ACCEPTED"
        or int(source_resolution_decision.get("terminal_record_count", 0)) != 20
        or int(source_resolution_decision.get("selected_B_interaction", 0)) != 999
        or source_resolution_decision.get("contract_hash")
        != source_resource_freeze.get("config_hash")
        or re.fullmatch(r"[0-9a-f]{64}", source_resolution_decision_sha256) is None
    ):
        raise ValueError(
            "source resolution decision is not the accepted B=999 decision"
        )

    source_completion_hash = str(
        source_development_completion.get("completion_sha256", "")
    )
    source_completion_identity = {
        key: value
        for key, value in source_development_completion.items()
        if key != "completion_sha256"
    }
    expected_completion_identity = {
        "status": "PHASE_COMPLETE",
        "phase": "development",
        "source_hash": source_resource_freeze.get("source_hash"),
        "config_hash": source_resource_freeze.get("config_hash"),
        "lock_hash": source_resource_freeze.get("lock_hash"),
    }
    if (
        re.fullmatch(r"[0-9a-f]{64}", source_completion_hash) is None
        or _stable_hash(source_completion_identity) != source_completion_hash
        or any(
            source_development_completion.get(key) != value
            for key, value in expected_completion_identity.items()
        )
        or float(source_development_completion.get("observed_total_au", -1.0)) < 0
    ):
        raise ValueError("source development completion is stale or invalid")

    selections_hash = _stable_hash(source_resource_freeze.get("selections"))
    adoption_basis = {
        "schema_version": 1,
        "status": "ACCEPTED_FOR_GENERATION_12_CONFIRMATION",
        "source_resource_freeze_sha256": source_freeze_hash,
        "source_resolution_decision_sha256": source_resolution_decision_sha256,
        "source_development_completion_sha256": source_completion_hash,
        "source_run_id": source_development_completion.get("run_id"),
        "target_run_id": target_run_id,
        "source_identity": {
            "source_hash": source_resource_freeze.get("source_hash"),
            "config_hash": source_resource_freeze.get("config_hash"),
            "lock_hash": source_resource_freeze.get("lock_hash"),
        },
        "target_identity": {
            "source_hash": target_source_hash,
            "config_hash": target_config_hash,
            "lock_hash": target_lock_hash,
        },
        "resource_selections_sha256": selections_hash,
        "telemetry_sha256": source_resource_freeze.get("telemetry_sha256"),
        "resource_decision_changed": False,
        "scientific_result_reused": "resolution_only",
        "excluded_prior_results": ["gate_b", "fixed_family_supplement"],
        "required_fresh_results": ["gate_b", "fixed_family_supplement"],
        "adoption_rule": (
            "reuse exact successful-pilot resource selections and accepted B=999 "
            "resolution only; execute fresh zero-overlap generation-12 confirmation"
        ),
    }
    adoption = {
        **adoption_basis,
        "generation_adoption_sha256": _stable_hash(adoption_basis),
    }

    freeze_identity = deepcopy(source_freeze_identity)
    freeze_identity.update(
        {
            "source_hash": target_source_hash,
            "config_hash": target_config_hash,
            "lock_hash": target_lock_hash,
            "generation_adoption_sha256": adoption["generation_adoption_sha256"],
            "source_resource_freeze_sha256": source_freeze_hash,
            "resource_selections_sha256": selections_hash,
            "resource_decision_changed": False,
        }
    )
    freeze = {
        **freeze_identity,
        "resource_freeze_sha256": _stable_hash(freeze_identity),
    }

    resolution_identity = {
        "operation": "resolution",
        "status": "completed",
        "decision": "ACCEPTED",
        "terminal_record_count": 20,
        "selected_B_interaction": 999,
        "contract_hash": target_config_hash,
        "adopted_resolution_decision_sha256": source_resolution_decision_sha256,
        "generation_adoption_sha256": adoption["generation_adoption_sha256"],
        "new_resolution_science_executed": False,
    }
    resolution = {
        **resolution_identity,
        "resolution_adoption_sha256": _stable_hash(resolution_identity),
    }

    completion_identity = deepcopy(source_completion_identity)
    completion_identity.update(
        {
            "run_id": target_run_id,
            "source_hash": target_source_hash,
            "config_hash": target_config_hash,
            "lock_hash": target_lock_hash,
            "source_completion_sha256": source_completion_hash,
            "generation_adoption_sha256": adoption["generation_adoption_sha256"],
            "scientific_result_reused": "resolution_only",
        }
    )
    completion = {
        **completion_identity,
        "completion_sha256": _stable_hash(completion_identity),
    }
    return freeze, resolution, completion, adoption


def write_generation_12_prerequisites(
    *,
    source_resource_freeze_path: Path,
    source_resolution_decision_path: Path,
    source_development_completion_path: Path,
    repo_root: Path,
    target_run_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Write immutable G12 prerequisite-adoption evidence for package generation."""
    from rfm_pipeline.campaign_contract import G11_CONTRACT, compute_contract_hash
    from rfm_pipeline.hpc_campaign_package import (
        _hash_file,
        _hash_python_tree,
        _validate_resource_freeze,
    )

    source_resource_freeze_path = source_resource_freeze_path.resolve()
    source_resolution_decision_path = source_resolution_decision_path.resolve()
    source_development_completion_path = source_development_completion_path.resolve()
    resolved_repo_root = repo_root.resolve()
    source_freeze = json.loads(source_resource_freeze_path.read_text(encoding="utf-8"))
    _validate_resource_freeze(source_freeze)
    resolution_bytes = source_resolution_decision_path.read_bytes()
    resolution = json.loads(resolution_bytes)
    completion = json.loads(
        source_development_completion_path.read_text(encoding="utf-8")
    )
    freeze, adopted_resolution, adopted_completion, adoption = (
        adopt_generation_12_prerequisites(
            source_resource_freeze=source_freeze,
            source_resolution_decision=resolution,
            source_resolution_decision_sha256=hashlib.sha256(
                resolution_bytes
            ).hexdigest(),
            source_development_completion=completion,
            target_source_hash=_hash_python_tree(
                resolved_repo_root / "src" / "rfm_pipeline"
            ),
            target_config_hash=compute_contract_hash(G11_CONTRACT),
            target_lock_hash=_hash_file(resolved_repo_root / "pixi.lock"),
            target_run_id=target_run_id,
        )
    )
    outputs = {
        "resource_freeze": output_dir.resolve() / "resource_freeze.json",
        "resolution_decision": output_dir.resolve() / "resolution_decision.json",
        "development_completion": (
            output_dir.resolve() / "development_completion.json"
        ),
        "generation_adoption": output_dir.resolve() / "generation_adoption.json",
    }
    for label, payload in (
        ("resource_freeze", freeze),
        ("resolution_decision", adopted_resolution),
        ("development_completion", adopted_completion),
        ("generation_adoption", adoption),
    ):
        _write_new_json(outputs[label], payload)
    return {
        "status": "GENERATION_12_PREREQUISITES_ADOPTED",
        "target_run_id": target_run_id,
        "outputs": {
            label: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for label, path in outputs.items()
        },
        "generation_adoption_sha256": adoption["generation_adoption_sha256"],
    }


def adopt_gate_b_decision_for_downstream(
    *,
    gate_b_decision: dict[str, Any],
    gate_b_decision_sha256: str,
    amendment: dict[str, Any],
) -> dict[str, Any]:
    """Carry a Gate-B PASS across the sole fixed-family-count amendment.

    Gate B does not read ``fixed_family_replicates``.  This bridge is valid
    only when that count is the amendment's sole changed field and the source
    decision has exact 5,600-task PASS coverage.
    """
    amendment_identity = {
        key: value
        for key, value in amendment.items()
        if key != "contract_amendment_sha256"
    }
    changed = amendment.get("changed_fields")
    old_hash = str(amendment.get("pilot_base_contract_hash", ""))
    new_hash = str(amendment.get("selected_contract_hash", ""))
    if (
        amendment.get("contract_amendment_sha256") != _stable_hash(amendment_identity)
        or changed != {"fixed_family_replicates": {"before": 1000, "after": 300}}
        or re.fullmatch(r"[0-9a-f]{64}", old_hash) is None
        or re.fullmatch(r"[0-9a-f]{64}", new_hash) is None
        or old_hash == new_hash
    ):
        raise ValueError("fixed-family contract amendment is stale or malformed")
    if (
        gate_b_decision.get("operation") != "gate_b"
        or gate_b_decision.get("status") != "completed"
        or gate_b_decision.get("decision") != "PASS"
        or gate_b_decision.get("contract_hash") != old_hash
        or int(gate_b_decision.get("terminal_record_count", 0)) != 5600
        or re.fullmatch(r"[0-9a-f]{64}", gate_b_decision_sha256) is None
    ):
        raise ValueError("Gate-B decision is not an exact passing source for adoption")
    identity = {
        "schema_version": 1,
        "operation": "gate_b",
        "status": "completed",
        "decision": "PASS",
        "contract_hash": new_hash,
        "terminal_record_count": 5600,
        "adoption_rule": "fixed_family_replicates_only_1000_to_300",
        "adopted_from_contract_hash": old_hash,
        "adopted_gate_b_decision_sha256": gate_b_decision_sha256,
        "contract_amendment_sha256": amendment["contract_amendment_sha256"],
    }
    return {**identity, "adoption_sha256": _stable_hash(identity)}


def write_gate_b_adoption(
    *,
    gate_b_decision_path: Path,
    amendment_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write the exact-byte-bound Gate-B prerequisite for downstream phases."""
    decision = json.loads(gate_b_decision_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    adopted = adopt_gate_b_decision_for_downstream(
        gate_b_decision=decision,
        gate_b_decision_sha256=hashlib.sha256(
            gate_b_decision_path.read_bytes()
        ).hexdigest(),
        amendment=amendment,
    )
    _write_new_json(output_path, adopted)
    return adopted


def build_development_admission_guard(
    *,
    stage_allocations: Any,
    resource_freeze: dict[str, Any],
    postprocessing_reserved_au: float,
    allocation_quota_au: float,
) -> dict[str, Any]:
    """Admit resolution only when its full request preserves the B=999 branch."""
    freeze_identity = {
        key: value
        for key, value in resource_freeze.items()
        if key != "resource_freeze_sha256"
    }
    if resource_freeze.get("resource_freeze_sha256") != _stable_hash(freeze_identity):
        raise ValueError("development admission requires a valid resource freeze")
    allocation = resource_freeze.get("allocation_accounting")
    if not isinstance(allocation, dict):
        raise TypeError("development admission lacks pilot allocation accounting")
    stages = list(stage_allocations)
    if len(stages) != 1:
        raise ValueError("development admission requires exactly one resolution stage")
    stage = stages[0]
    stage_name = str(
        stage["stage_name"] if isinstance(stage, dict) else stage.stage_name
    )
    requested_au = float(
        stage["requested_au"] if isinstance(stage, dict) else stage.requested_au
    )
    if (
        stage_name != "resolution"
        or not math.isfinite(requested_au)
        or requested_au < 0
    ):
        raise ValueError(
            "development admission requires one valid resolution allocation"
        )
    prior_au = float(allocation.get("prior_rejected_attempt_au", -1.0))
    pilot_au = float(allocation.get("accepted_pilot_observed_au", -1.0))
    values = (
        prior_au,
        pilot_au,
        float(postprocessing_reserved_au),
        float(allocation_quota_au),
    )
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("development admission allocation values are invalid")
    whole_campaign_maximum_au = (
        prior_au
        + pilot_au
        + requested_au
        + PUBLICATION_B999_CONFIRMATORY_REQUESTED_AU
        + float(postprocessing_reserved_au)
    )
    if whole_campaign_maximum_au > float(allocation_quota_au) + 1.0e-9:
        raise ValueError(
            "development request would eliminate the affordable B=999 branch: "
            f"{whole_campaign_maximum_au:.6f} > {float(allocation_quota_au):.6f}"
        )
    identity = {
        "schema_version": 1,
        "status": "DEVELOPMENT_WITHIN_ALLOCATION",
        "resource_freeze_sha256": resource_freeze["resource_freeze_sha256"],
        "prior_rejected_attempt_au": prior_au,
        "accepted_pilot_observed_au": pilot_au,
        "development_requested_au": requested_au,
        "b999_confirmatory_estimated_au": (PUBLICATION_B999_CONFIRMATORY_ESTIMATED_AU),
        "b999_confirmatory_requested_au_with_shared_reserve": (
            PUBLICATION_B999_CONFIRMATORY_REQUESTED_AU
        ),
        "postprocessing_reserved_au": float(postprocessing_reserved_au),
        "whole_campaign_maximum_au": whole_campaign_maximum_au,
        "allocation_quota_au": float(allocation_quota_au),
        "headroom_au": float(allocation_quota_au) - whole_campaign_maximum_au,
    }
    return {**identity, "development_admission_sha256": _stable_hash(identity)}


def build_campaign_budget_certificate(
    *,
    run_id: str,
    campaign_requested_au: float,
    prior_rejected_attempt_au: float,
    postprocessing_reserved_au: float,
    allocation_quota_au: float,
    resource_freeze_sha256: str,
    campaign_inventory_hash: str,
    contract_amendment_sha256: str,
    accepted_pilot_observed_au: float | None = None,
    development_resolution_observed_au: float | None = None,
    remaining_confirmatory_estimated_au: float | None = None,
    remaining_confirmatory_requested_au_with_shared_reserve: float | None = None,
) -> dict[str, Any]:
    """Certify the conservative whole-campaign projection against the hard cap.

    When the detailed components are supplied, completed pilot and resolution
    estimates are replaced by their exact observed ``sacct`` AUs and the shared
    20% reserve is applied only to unexecuted confirmatory work. Rejected
    attempts and the bounded final artifact-compilation job are added separately.
    """
    values = (
        campaign_requested_au,
        prior_rejected_attempt_au,
        postprocessing_reserved_au,
        allocation_quota_au,
    )
    if any(float(value) < 0 for value in values):
        raise ValueError("allocation values must be nonnegative")
    if re.fullmatch(r"[0-9a-f]{64}", contract_amendment_sha256) is None:
        raise ValueError("contract amendment hash is malformed")
    components = (
        accepted_pilot_observed_au,
        development_resolution_observed_au,
        remaining_confirmatory_estimated_au,
        remaining_confirmatory_requested_au_with_shared_reserve,
    )
    if any(value is None for value in components):
        raise ValueError(
            "budget certification requires exact completed-phase accounting"
        )
    if any(float(value) < 0 for value in components):
        raise ValueError("campaign allocation components must be nonnegative")
    expected_remaining_requested_au = math.ceil(
        float(remaining_confirmatory_estimated_au) * 1.20
    )
    if not math.isclose(
        float(remaining_confirmatory_requested_au_with_shared_reserve),
        expected_remaining_requested_au,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("remaining confirmatory reserve is not exactly 20 percent")
    component_total = (
        float(accepted_pilot_observed_au)
        + float(development_resolution_observed_au)
        + float(remaining_confirmatory_requested_au_with_shared_reserve)
    )
    if not math.isclose(
        float(campaign_requested_au), component_total, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError("campaign allocation components do not sum to requested AUs")
    total = (
        float(campaign_requested_au)
        + float(prior_rejected_attempt_au)
        + float(postprocessing_reserved_au)
    )
    if total > float(allocation_quota_au) + 1.0e-9:
        raise ValueError(
            "whole-campaign projection exceeds the 25,000-AU ceiling: "
            f"{total:.6f} > {float(allocation_quota_au):.6f}"
        )
    identity = {
        "schema_version": 1,
        "status": "WITHIN_ALLOCATION",
        "run_id": run_id,
        "campaign_requested_au_including_shared_reserve": float(campaign_requested_au),
        "prior_rejected_attempt_au": float(prior_rejected_attempt_au),
        "postprocessing_reserved_au": float(postprocessing_reserved_au),
        "whole_campaign_projected_au": total,
        "allocation_quota_au": float(allocation_quota_au),
        "headroom_au": float(allocation_quota_au) - total,
        "resource_freeze_sha256": resource_freeze_sha256,
        "campaign_inventory_hash": campaign_inventory_hash,
        "contract_amendment_sha256": contract_amendment_sha256,
    }
    development_resolution_au_ceiling = (
        float(allocation_quota_au)
        - float(prior_rejected_attempt_au)
        - float(postprocessing_reserved_au)
        - float(accepted_pilot_observed_au)
        - float(remaining_confirmatory_requested_au_with_shared_reserve)
    )
    identity.update(
        {
            "accepted_pilot_observed_au": float(accepted_pilot_observed_au),
            "development_resolution_observed_au": float(
                development_resolution_observed_au
            ),
            "remaining_confirmatory_estimated_au": float(
                remaining_confirmatory_estimated_au
            ),
            "remaining_confirmatory_requested_au_with_shared_reserve": float(
                remaining_confirmatory_requested_au_with_shared_reserve
            ),
            "development_resolution_au_ceiling_for_selected_design": (
                development_resolution_au_ceiling
            ),
            "shared_reserve_fraction_on_unexecuted_confirmatory_work": 0.20,
        }
    )
    return {**identity, "budget_certificate_sha256": _stable_hash(identity)}


def build_downstream_budget_certificate(
    *,
    run_id: str,
    completed_observed_au: float,
    remaining_downstream_estimated_au: float,
    postprocessing_reserved_au: float,
    allocation_quota_au: float,
    resource_freeze_sha256: str,
    campaign_inventory_hash: str,
    contract_amendment_sha256: str,
) -> dict[str, Any]:
    """Certify a downstream-only package from exact completed campaign spend."""
    values = (
        completed_observed_au,
        remaining_downstream_estimated_au,
        postprocessing_reserved_au,
        allocation_quota_au,
    )
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
        raise ValueError("downstream allocation values must be finite and nonnegative")
    if float(allocation_quota_au) <= 0:
        raise ValueError("allocation quota must be positive")
    for label, value in (
        ("resource freeze", resource_freeze_sha256),
        ("campaign inventory", campaign_inventory_hash),
        ("contract amendment", contract_amendment_sha256),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{label} hash is malformed")
    remaining_requested_au = math.ceil(float(remaining_downstream_estimated_au) * 1.20)
    total = (
        float(completed_observed_au)
        + float(remaining_requested_au)
        + float(postprocessing_reserved_au)
    )
    if total > float(allocation_quota_au) + 1.0e-9:
        raise ValueError(
            "whole-campaign projection exceeds the allocation ceiling: "
            f"{total:.6f} > {float(allocation_quota_au):.6f}"
        )
    identity = {
        "schema_version": 1,
        "status": "DOWNSTREAM_WITHIN_ALLOCATION",
        "run_id": run_id,
        "completed_observed_au": float(completed_observed_au),
        "remaining_downstream_estimated_au": float(remaining_downstream_estimated_au),
        "remaining_downstream_requested_au_with_shared_reserve": float(
            remaining_requested_au
        ),
        "shared_reserve_fraction_on_unexecuted_downstream_work": 0.20,
        "postprocessing_reserved_au": float(postprocessing_reserved_au),
        "whole_campaign_projected_au": total,
        "allocation_quota_au": float(allocation_quota_au),
        "headroom_au": float(allocation_quota_au) - total,
        "resource_freeze_sha256": resource_freeze_sha256,
        "campaign_inventory_hash": campaign_inventory_hash,
        "contract_amendment_sha256": contract_amendment_sha256,
    }
    return {**identity, "budget_certificate_sha256": _stable_hash(identity)}


def _remaining_confirmatory_estimated_au(stage_allocations: Any) -> float:
    """Return estimated AUs only for stages not completed before final packaging."""
    remaining = 0.0
    for stage in stage_allocations:
        stage_name = str(
            stage["stage_name"] if isinstance(stage, dict) else stage.stage_name
        )
        if stage_name in {
            "scheduler_diagnostic",
            "resolution",
        } or stage_name.startswith("pilot_"):
            continue
        estimated = (
            stage["estimated_au"] if isinstance(stage, dict) else stage.estimated_au
        )
        remaining += float(estimated)
    return remaining


def validate_phase_budget_guard(
    *,
    stage_allocations: Any,
    completed_stage_names: set[str],
    current_stage_names: set[str],
    prior_rejected_attempt_au: float,
    accepted_pilot_observed_au: float,
    development_resolution_observed_au: float,
    completed_confirmatory_observed_au: float,
    postprocessing_reserved_au: float,
    allocation_quota_au: float,
) -> dict[str, Any]:
    """Require both the rolling projection and current-phase maximum to fit."""
    allocations: dict[str, tuple[float, float]] = {}
    for stage in stage_allocations:
        name = str(stage["stage_name"] if isinstance(stage, dict) else stage.stage_name)
        if name in {"scheduler_diagnostic", "resolution"} or name.startswith("pilot_"):
            continue
        estimated = float(
            stage["estimated_au"] if isinstance(stage, dict) else stage.estimated_au
        )
        requested = float(
            stage["requested_au"] if isinstance(stage, dict) else stage.requested_au
        )
        if (
            name in allocations
            or not math.isfinite(estimated)
            or not math.isfinite(requested)
            or estimated < 0
            or requested < estimated
        ):
            raise ValueError("campaign stage allocation is duplicate or invalid")
        allocations[name] = (estimated, requested)
    known = set(allocations)
    if (
        not current_stage_names
        or not completed_stage_names <= known
        or not current_stage_names <= known
        or completed_stage_names & current_stage_names
    ):
        raise ValueError("phase budget stage coverage is invalid")
    actual_values = (
        prior_rejected_attempt_au,
        accepted_pilot_observed_au,
        development_resolution_observed_au,
        completed_confirmatory_observed_au,
        postprocessing_reserved_au,
        allocation_quota_au,
    )
    if any(not math.isfinite(value) or value < 0 for value in actual_values):
        raise ValueError(
            "phase budget allocation values must be finite and nonnegative"
        )
    actual_spent = (
        prior_rejected_attempt_au
        + accepted_pilot_observed_au
        + development_resolution_observed_au
        + completed_confirmatory_observed_au
    )
    remaining_estimated_au = sum(
        estimated
        for name, (estimated, _requested) in allocations.items()
        if name not in completed_stage_names
    )
    remaining_reserved_au = math.ceil(remaining_estimated_au * 1.20)
    rolling_projected_au = (
        actual_spent + remaining_reserved_au + postprocessing_reserved_au
    )
    if rolling_projected_au > allocation_quota_au + 1.0e-9:
        raise ValueError(
            "rolling whole-campaign projection exceeds the 25,000-AU ceiling: "
            f"{rolling_projected_au:.6f} > {allocation_quota_au:.6f}"
        )
    current_phase_requested_au = sum(
        allocations[name][1] for name in current_stage_names
    )
    current_phase_maximum_au = (
        actual_spent + current_phase_requested_au + postprocessing_reserved_au
    )
    if current_phase_maximum_au > allocation_quota_au + 1.0e-9:
        raise ValueError(
            "current phase could cross the 25,000-AU ceiling at requested walltime: "
            f"{current_phase_maximum_au:.6f} > {allocation_quota_au:.6f}"
        )
    identity = {
        "schema_version": 1,
        "status": "PHASE_WITHIN_ALLOCATION",
        "completed_stage_names": sorted(completed_stage_names),
        "current_stage_names": sorted(current_stage_names),
        "actual_spent_au": actual_spent,
        "remaining_estimated_au": remaining_estimated_au,
        "remaining_requested_au_with_shared_reserve": remaining_reserved_au,
        "rolling_whole_campaign_projected_au": rolling_projected_au,
        "current_phase_requested_au": current_phase_requested_au,
        "current_phase_maximum_au": current_phase_maximum_au,
        "postprocessing_reserved_au": postprocessing_reserved_au,
        "allocation_quota_au": allocation_quota_au,
    }
    return {**identity, "phase_budget_guard_sha256": _stable_hash(identity)}


def _validate_development_completion(
    payload: dict[str, Any],
    *,
    expected_run_id: str,
    resource_freeze: dict[str, Any],
) -> float:
    """Return exact resolution AUs from one self-hashed completion record."""
    identity = {
        key: value for key, value in payload.items() if key != "completion_sha256"
    }
    observed_au = float(payload.get("observed_total_au", -1.0))
    expected = {
        "status": "PHASE_COMPLETE",
        "phase": "development",
        "run_id": expected_run_id,
        "source_hash": resource_freeze.get("source_hash"),
        "config_hash": resource_freeze.get("config_hash"),
        "lock_hash": resource_freeze.get("lock_hash"),
    }
    if (
        payload.get("completion_sha256") != _stable_hash(identity)
        or any(payload.get(field) != value for field, value in expected.items())
        or observed_au < 0
    ):
        raise ValueError("development completion accounting is stale or invalid")
    return observed_au


def close_completed_pilot(
    *,
    dag: Any,
    submission_record_path: Path,
    accepted_resource_freeze_path: Path | None = None,
    accounting_output_dir: Path,
    resource_freeze_path: Path,
    prior_sunk_au: float,
    run_command: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect exact Slurm evidence and write one allocation-bound resource freeze."""
    from rfm_pipeline.hpc_campaign_package import (
        build_submission_plan,
        collect_pilot_accounting,
        select_pilot_resources,
    )

    plan = build_submission_plan(dag)
    expected_steps = {str(step["step_id"]) for step in plan["pilot_steps"]}
    job_ids = validate_completed_submission_record(
        submission_record_path,
        expected_step_ids=expected_steps,
        expected_identity={
            "run_id": dag.run_id,
            "source_hash": dag.source_hash,
            "config_hash": dag.config_hash,
            "lock_hash": dag.lock_hash,
        },
    )
    accounting = collect_pilot_accounting(
        dag,
        submission_job_ids=job_ids,
        output_dir=accounting_output_dir,
        run_command=run_command,
    )
    freeze = select_pilot_resources(
        pilot_matrix=dag.pilot_matrix,
        telemetry=accounting["pilot_telemetry"],
        cluster=dag.cluster,
        source_hash=dag.source_hash,
        config_hash=dag.config_hash,
        lock_hash=dag.lock_hash,
    )
    freeze = augment_resource_freeze_accounting(
        freeze,
        pilot_accounting=accounting,
        prior_sunk_au=prior_sunk_au,
    )
    if accepted_resource_freeze_path is not None:
        accepted = json.loads(
            accepted_resource_freeze_path.resolve().read_text(encoding="utf-8")
        )
        freeze = _reconcile_accepted_resource_freeze(freeze, accepted)
    _write_new_json(resource_freeze_path, freeze)
    return accounting, freeze


def load_packaged_dag(
    *, package_root: str | Path, config_path: str | Path, repo_root: str | Path
) -> Any:
    """Reconstruct and verify an immutable package without regenerating it.

    Regeneration is unsafe for completed work because package paths contribute
    to manifest identity.  This loader uses the package's own manifests and
    summary, verifies their hashes/identities, and returns the native RFM DAG
    object needed by accounting, preflight, and submission functions.
    """
    from rfm_pipeline.hpc_campaign_package import (
        CampaignDAG,
        CampaignEnvelope,
        ResourceEnvelope,
        StageAllocationEstimate,
        StagePlan,
        _cluster_from_mapping,
        _load_campaign_config,
        load_stage_manifest,
    )

    package = Path(package_root).resolve()
    summary = json.loads((package / "package_summary.json").read_text(encoding="utf-8"))
    raw_config = _load_campaign_config(Path(config_path).resolve())
    # The immutable package is authoritative for run identity.  The canonical
    # campaign config is supplied only to reconstruct the Kestrel cluster
    # policy; pilot packages intentionally use unique run IDs without mutating
    # the tracked canonical config for every attempt.
    cluster = _cluster_from_mapping(raw_config["cluster"])
    inventory_path = Path(str(summary["campaign_inventory_path"]))
    if not inventory_path.is_file() or _sha256_path(inventory_path) != summary.get(
        "campaign_inventory_hash"
    ):
        raise ValueError("packaged campaign inventory is absent or hash-mismatched")

    raw_allocations = summary["campaign_envelope"]["stage_allocations"]
    envelope_values = dict(summary["campaign_envelope"])
    envelope_values["stage_allocations"] = tuple(
        StageAllocationEstimate(**row) for row in raw_allocations
    )
    envelope = CampaignEnvelope(**envelope_values)

    stage_rows = list(summary.get("stages", []))
    reducer_stage_by_hash = {
        str(row["reducer_output_hash"]): str(row["name"]) for row in stage_rows
    }
    stages = []
    for row in stage_rows:
        manifest_path = Path(str(row["manifest_path"]))
        records = load_stage_manifest(manifest_path)
        if len(records) != int(row["job_count"]):
            raise ValueError(f"packaged stage {row['name']} job count differs")
        if any(
            record.get("run_id") != summary["run_id"]
            or record.get("source_hash") != summary["source_hash"]
            or record.get("config_hash") != summary["config_hash"]
            or record.get("lock_hash") != summary["lock_hash"]
            for record in records
        ):
            raise ValueError(f"packaged stage {row['name']} identity differs")
        parent_hashes = {str(record["parent_hash"]) for record in records}
        if len(parent_hashes) != 1:
            raise ValueError(f"packaged stage {row['name']} mixes parent identities")
        parent_name = reducer_stage_by_hash.get(parent_hashes.pop())
        output_root = Path(str(records[0]["output_dir"])).parent
        worker_resources = ResourceEnvelope(**records[0]["worker_resources"])
        stage_package_dir = manifest_path.parent
        stages.append(
            StagePlan(
                name=str(row["name"]),
                partition=str(row["partition"]),
                job_count=int(row["job_count"]),
                parent_stage_name=parent_name,
                worker_resources=worker_resources,
                reducer_resources=worker_resources,
                manifest_path=manifest_path,
                worker_script_path=Path(str(row["worker_script_path"])),
                worker_script_paths=tuple(
                    Path(value) for value in row["worker_script_paths"]
                ),
                audit_script_path=Path(str(row["audit_script_path"])),
                audit_output_dir=stage_package_dir / "audit",
                reducer_script_path=Path(str(row["reducer_script_path"])),
                output_root=output_root,
                reducer_output_dir=output_root.parent / "reducer",
                reducer_output_hash=str(row["reducer_output_hash"]),
                reducer_expected_range=(0, int(row["job_count"])),
            )
        )
    if not stages:
        raise ValueError("campaign package contains no stages")

    blockers = json.loads(
        (package / "readiness_blockers.json").read_text(encoding="utf-8")
    )
    telemetry = json.loads(
        (package / "telemetry_schema.json").read_text(encoding="utf-8")
    )
    pilot_matrix = json.loads(
        (package / "pilot_matrix.json").read_text(encoding="utf-8")
    )
    post_pilot = json.loads(
        (package / "post_pilot_selection.json").read_text(encoding="utf-8")
    )
    dag = CampaignDAG(
        run_id=str(summary["run_id"]),
        package_mode=str(summary["package_mode"]),
        repo_root=Path(repo_root).resolve(),
        output_dir=package,
        cluster=cluster,
        scheduler_submission_permitted=bool(summary["scheduler_submission_permitted"]),
        source_hash=str(summary["source_hash"]),
        config_hash=str(summary["config_hash"]),
        lock_hash=str(summary["lock_hash"]),
        contract_config_path=package / "contract" / "g11_campaign_contract.toml",
        campaign_inventory_path=inventory_path,
        campaign_inventory_hash=str(summary["campaign_inventory_hash"]),
        readiness_blockers=tuple(str(value) for value in blockers["blockers"]),
        telemetry_schema=tuple(telemetry),
        pilot_matrix=tuple(pilot_matrix),
        post_pilot_selection=post_pilot,
        campaign_envelope=envelope,
        stages=tuple(stages),
    )
    plan = json.loads((package / "submission_plan.json").read_text(encoding="utf-8"))
    identity = {
        key: value for key, value in plan.items() if key != "submission_plan_sha256"
    }
    if plan.get("submission_plan_sha256") != _stable_hash(identity):
        raise ValueError("packaged submission plan self-hash differs")
    if any(
        plan.get(field) != getattr(dag, field)
        for field in (
            "run_id",
            "source_hash",
            "config_hash",
            "lock_hash",
            "campaign_inventory_hash",
        )
    ):
        raise ValueError("packaged submission plan identity differs")
    return dag


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_development_package(
    *,
    output_dir: Path,
    resource_freeze_path: Path,
    repo_root: Path,
    config_path: Path,
) -> Any:
    """Build the resolution-only source package after accepted pilot accounting."""
    from rfm_pipeline.hpc_campaign_package import generate_campaign_package

    freeze = json.loads(resource_freeze_path.read_text(encoding="utf-8"))
    if "allocation_accounting" not in freeze:
        raise ValueError("resource freeze lacks whole-campaign allocation accounting")
    return generate_campaign_package(
        output_dir=output_dir,
        repo_root=repo_root,
        config_path=config_path,
        resource_freeze=freeze,
        package_mode="development",
    )


def prepare_final_package(
    *,
    output_dir: Path,
    resource_freeze_path: Path,
    resolution_decision_path: Path,
    development_completion_path: Path,
    repo_root: Path,
    config_path: Path,
    allocation_quota_au: float,
    postprocessing_reserved_au: float,
    budget_certificate_path: Path,
) -> tuple[Any, dict[str, Any]]:
    """Build and budget-certify the only package permitted for final science."""
    from rfm_pipeline.campaign_contract import G11_CONTRACT, compute_contract_hash
    from rfm_pipeline.hpc_campaign_package import (
        _hash_file,
        _hash_python_tree,
        _load_campaign_config,
        _validate_resource_freeze,
        generate_campaign_package,
    )

    freeze = json.loads(resource_freeze_path.read_text(encoding="utf-8"))
    _validate_resource_freeze(freeze)
    allocation = freeze.get("allocation_accounting")
    if not isinstance(allocation, dict):
        raise TypeError("resource freeze lacks allocation accounting")
    resolved_repo = repo_root.resolve()
    pilot_base_hash = compute_contract_hash(G11_CONTRACT)
    if (
        freeze.get("source_hash")
        != _hash_python_tree(resolved_repo / "src" / "rfm_pipeline")
        or freeze.get("lock_hash") != _hash_file(resolved_repo / "pixi.lock")
        or freeze.get("config_hash") != pilot_base_hash
    ):
        raise ValueError(
            "resource freeze does not belong to the accepted source, lock, and pilot base contract"
        )
    decision = json.loads(resolution_decision_path.read_text(encoding="utf-8"))
    if (
        decision.get("operation") != "resolution"
        or decision.get("status") != "completed"
        or decision.get("decision") != "ACCEPTED"
        or int(decision.get("terminal_record_count", 0)) != 20
        or decision.get("contract_hash") != pilot_base_hash
    ):
        raise ValueError(
            "resolution decision is not valid for the accepted pilot base contract"
        )
    selected_b = int(decision.get("selected_B_interaction", 0))
    if selected_b not in {
        G11_CONTRACT.resolution_base_draws,
        2 * G11_CONTRACT.resolution_base_draws,
    }:
        raise ValueError(
            "resolution decision selected a non-prespecified interaction schedule"
        )
    expected_run_id = str(_load_campaign_config(config_path)["run_id"])
    development_completion = json.loads(
        development_completion_path.read_text(encoding="utf-8")
    )
    development_observed_au = _validate_development_completion(
        development_completion,
        expected_run_id=expected_run_id,
        resource_freeze=freeze,
    )
    publication_base, amendment = build_publication_contract_amendment(G11_CONTRACT)
    decision_sha256 = _hash_file(resolution_decision_path.resolve())
    selected_contract = replace(
        publication_base,
        B_interaction=selected_b,
        resolution_decision_sha256=decision_sha256,
    )
    accepted_pilot_observed_au = float(
        allocation.get("accepted_pilot_observed_au", -1.0)
    )
    prior_rejected_attempt_au = float(allocation.get("prior_rejected_attempt_au", -1.0))
    if accepted_pilot_observed_au < 0:
        raise ValueError("resource freeze lacks accepted-pilot observed allocation")
    if prior_rejected_attempt_au < 0:
        raise ValueError("resource freeze lacks prior rejected-attempt allocation")
    dag = generate_campaign_package(
        output_dir=output_dir,
        resource_freeze=freeze,
        contract=selected_contract,
        package_mode="confirmatory",
        repo_root=resolved_repo,
        config_path=config_path,
        completed_observed_au_for_admission=(
            accepted_pilot_observed_au
            + prior_rejected_attempt_au
            + development_observed_au
        ),
        postprocessing_reserved_au_for_admission=postprocessing_reserved_au,
    )
    amendment_identity = {
        key: value
        for key, value in amendment.items()
        if key != "contract_amendment_sha256"
    }
    amendment_identity.update(
        {
            "resolution_decision_sha256": decision_sha256,
            "selected_B_interaction": selected_b,
            "selected_contract_hash": dag.config_hash,
        }
    )
    amendment = {
        **amendment_identity,
        "contract_amendment_sha256": _stable_hash(amendment_identity),
    }
    _write_new_json(
        output_dir.resolve() / "contract" / "fixed_family_amendment.json",
        amendment,
    )
    remaining_estimated_au = _remaining_confirmatory_estimated_au(
        dag.campaign_envelope.stage_allocations
    )
    remaining_requested_au = math.ceil(remaining_estimated_au * 1.20)
    adjusted_campaign_requested_au = (
        accepted_pilot_observed_au + development_observed_au + remaining_requested_au
    )
    if dag.run_id != expected_run_id:
        raise ValueError(
            "development completion run identity differs from final package"
        )
    certificate = build_campaign_budget_certificate(
        run_id=dag.run_id,
        campaign_requested_au=adjusted_campaign_requested_au,
        prior_rejected_attempt_au=prior_rejected_attempt_au,
        postprocessing_reserved_au=postprocessing_reserved_au,
        allocation_quota_au=allocation_quota_au,
        resource_freeze_sha256=str(freeze["resource_freeze_sha256"]),
        campaign_inventory_hash=dag.campaign_inventory_hash,
        contract_amendment_sha256=amendment["contract_amendment_sha256"],
        accepted_pilot_observed_au=accepted_pilot_observed_au,
        development_resolution_observed_au=development_observed_au,
        remaining_confirmatory_estimated_au=remaining_estimated_au,
        remaining_confirmatory_requested_au_with_shared_reserve=remaining_requested_au,
    )
    _write_new_json(budget_certificate_path, certificate)
    return dag, certificate


def prepare_downstream_package(
    *,
    output_dir: Path,
    resource_freeze_path: Path,
    gate_b_contract_path: Path,
    repo_root: Path,
    config_path: Path,
    completed_observed_au: float,
    allocation_quota_au: float,
    postprocessing_reserved_au: float,
    budget_certificate_path: Path,
) -> tuple[Any, dict[str, Any]]:
    """Build the fixed-family-and-later package without repeating Gate B."""
    from rfm_pipeline.campaign_contract import load_contract
    from rfm_pipeline.hpc_campaign_package import (
        _hash_file,
        _hash_python_tree,
        _load_campaign_config,
        _validate_resource_freeze,
        generate_campaign_package,
    )

    freeze = json.loads(resource_freeze_path.read_text(encoding="utf-8"))
    _validate_resource_freeze(freeze)
    resolved_repo = repo_root.resolve()
    if freeze.get("source_hash") != _hash_python_tree(
        resolved_repo / "src" / "rfm_pipeline"
    ) or freeze.get("lock_hash") != _hash_file(resolved_repo / "pixi.lock"):
        raise ValueError(
            "resource freeze does not belong to the accepted source and lock"
        )
    base_contract, base_hash = load_contract(gate_b_contract_path.resolve())
    if base_contract.resolution_decision_sha256 == "PENDING":
        raise ValueError("Gate-B contract lacks the accepted resolution decision")
    raw_config = _load_campaign_config(config_path.resolve())
    configured_quota = float(raw_config["cluster"]["allocation_quota"])
    if not math.isclose(
        configured_quota,
        float(allocation_quota_au),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            "downstream budget ceiling differs from packaged scheduler configuration"
        )
    selected_contract, amendment = build_publication_contract_amendment(base_contract)
    dag = generate_campaign_package(
        output_dir=output_dir,
        resource_freeze=freeze,
        contract=selected_contract,
        package_mode="downstream",
        repo_root=resolved_repo,
        config_path=config_path,
        completed_observed_au_for_admission=float(completed_observed_au),
        postprocessing_reserved_au_for_admission=float(postprocessing_reserved_au),
    )
    amendment_identity = {
        key: value
        for key, value in amendment.items()
        if key != "contract_amendment_sha256"
    }
    amendment_identity.update(
        {
            "gate_b_source_contract_hash": base_hash,
            "selected_contract_hash": dag.config_hash,
        }
    )
    amendment = {
        **amendment_identity,
        "contract_amendment_sha256": _stable_hash(amendment_identity),
    }
    _write_new_json(
        output_dir.resolve() / "contract" / "fixed_family_amendment.json",
        amendment,
    )
    remaining_estimated_au = sum(
        float(item.estimated_au) for item in dag.campaign_envelope.stage_allocations
    )
    certificate = build_downstream_budget_certificate(
        run_id=dag.run_id,
        completed_observed_au=float(completed_observed_au),
        remaining_downstream_estimated_au=remaining_estimated_au,
        postprocessing_reserved_au=float(postprocessing_reserved_au),
        allocation_quota_au=float(allocation_quota_au),
        resource_freeze_sha256=str(freeze["resource_freeze_sha256"]),
        campaign_inventory_hash=dag.campaign_inventory_hash,
        contract_amendment_sha256=amendment["contract_amendment_sha256"],
    )
    _write_new_json(budget_certificate_path, certificate)
    return dag, certificate


def _validate_phase_manifest_authorization_paths(
    dag: Any, *, phase: str, authorization_path: Path
) -> int:
    """Require every target-phase work unit to bind the authorization being written."""
    expected = authorization_path.resolve()
    matched = 0
    for stage in dag.stages:
        manifest_path = Path(stage.manifest_path)
        for line_number, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            fixed_family_alias = (
                phase == "fixed_family"
                and record.get("stage") == "fixed_family_supplement"
            )
            if record.get("phase") != phase and not fixed_family_alias:
                continue
            matched += 1
            observed_raw = record.get("execution_authorization_path")
            if (
                not isinstance(observed_raw, str)
                or Path(observed_raw).resolve() != expected
            ):
                raise ValueError(
                    "phase manifest authorization path differs from the authorization "
                    f"target: {manifest_path}:{line_number}"
                )
    if matched == 0:
        raise ValueError(f"phase {phase} has no manifest-bound scientific work units")
    return matched


def preflight_and_authorize_phase(
    *,
    dag: Any,
    phase: str,
    remaining_au: int,
    evidence_dir: Path,
    resource_freeze_path: Path,
    prerequisite_paths: tuple[Path, ...],
    authorization_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run same-day probes and write the exact phase authorization."""
    from rfm_pipeline.hpc_campaign_package import (
        run_hpc_live_smoke,
        write_phase_authorization,
    )

    _validate_phase_manifest_authorization_paths(
        dag, phase=phase, authorization_path=authorization_path
    )
    result = run_hpc_live_smoke(
        dag,
        target_phase=phase,
        remaining_au=remaining_au,
        evidence_dir=evidence_dir,
    )
    preflight_path = evidence_dir / "preflight.json"
    authorization = write_phase_authorization(
        dag,
        phase=phase,
        resource_freeze_path=resource_freeze_path,
        preflight_path=preflight_path,
        prerequisite_paths=prerequisite_paths,
        output_path=authorization_path,
    )
    return result["preflight"], authorization


def submit_authorized_phase(
    *,
    dag: Any,
    phase: str,
    preflight_path: Path,
    authorization_path: Path,
    submission_record_path: Path,
) -> dict[str, str]:
    """Submit one independently gated phase and persist every returned job ID."""
    from rfm_pipeline.hpc_campaign_package import (
        build_campaign_phase_plan,
        execute_campaign_phase_plan,
    )

    plan = build_campaign_phase_plan(dag, phase=phase)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    journal_path = submission_record_path.with_suffix(".submission_journal.jsonl")
    abort_path = submission_record_path.with_suffix(".submission_abort.json")
    if journal_path.exists() or abort_path.exists():
        raise ValueError(
            "phase submission evidence already exists; refusing to risk duplicate jobs"
        )
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    submitted_index = 0

    def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal submitted_index
        steps = plan["steps"]
        if submitted_index >= len(steps):
            raise RuntimeError(
                "phase submitter issued more scheduler calls than plan steps"
            )
        step_id = str(steps[submitted_index]["step_id"])
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=_scheduler_environment(),
        )
        raw = str(completed.stdout).strip()
        job_id = raw.split(";", maxsplit=1)[0] if raw else None
        entry = {
            "sequence": submitted_index,
            "step_id": step_id,
            "command": [str(value) for value in command],
            "returncode": int(completed.returncode),
            "job_id": job_id if job_id and re.fullmatch(r"[0-9]+", job_id) else None,
            "stdout": str(completed.stdout),
            "stderr": str(completed.stderr),
        }
        with journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        submitted_index += 1
        return completed

    try:
        job_ids = execute_campaign_phase_plan(
            plan,
            run_command=_run,
            authorize=True,
            preflight=preflight,
            phase_authorization=authorization,
        )
        if submitted_index != len(plan["steps"]):
            raise RuntimeError(
                "phase submitter did not execute exactly one call per plan step"
            )
    except Exception as exc:
        submitted_job_ids: list[str] = []
        if journal_path.is_file():
            for line in journal_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                job_id = row.get("job_id")
                if isinstance(job_id, str) and re.fullmatch(r"[0-9]+", job_id):
                    submitted_job_ids.append(job_id)
        cancellation = None
        if submitted_job_ids:
            cancellation = subprocess.run(
                ["scancel", "--quiet", *submitted_job_ids],
                capture_output=True,
                text=True,
                check=False,
                env=_scheduler_environment(),
            )
        abort_identity = {
            "schema_version": 1,
            "status": "SUBMISSION_ABORTED",
            "phase": phase,
            "submitted_job_ids": submitted_job_ids,
            "cancellation_returncode": (
                int(cancellation.returncode) if cancellation is not None else None
            ),
            "cancellation_stdout": (
                str(cancellation.stdout) if cancellation is not None else ""
            ),
            "cancellation_stderr": (
                str(cancellation.stderr) if cancellation is not None else ""
            ),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        _write_new_json(
            abort_path,
            {
                **abort_identity,
                "submission_abort_sha256": _stable_hash(abort_identity),
            },
        )
        if cancellation is not None and cancellation.returncode != 0:
            raise RuntimeError(
                "phase submission failed and submitted-job cancellation also failed"
            ) from exc
        raise
    identity = {
        "schema_version": 1,
        "status": "SUBMITTED",
        "phase": phase,
        "run_id": dag.run_id,
        "source_hash": dag.source_hash,
        "config_hash": dag.config_hash,
        "lock_hash": dag.lock_hash,
        "campaign_inventory_hash": dag.campaign_inventory_hash,
        "preflight_sha256": preflight["preflight_sha256"],
        "authorization_sha256": authorization["authorization_sha256"],
        "submission_plan_sha256": plan["submission_plan_sha256"],
        "submission_journal_sha256": _sha256_path(journal_path),
        "submission_job_ids": job_ids,
    }
    _write_new_json(
        submission_record_path,
        {**identity, "submission_record_sha256": _stable_hash(identity)},
    )
    return job_ids


def verify_phase_completion(
    *,
    dag: Any,
    phase: str,
    submission_record_path: Path,
    evidence_dir: Path,
    completion_path: Path,
    run_command: Any | None = None,
) -> dict[str, Any]:
    """Verify scheduler and reducer evidence before any downstream phase."""
    from rfm_pipeline.hpc_campaign_package import build_campaign_phase_plan

    plan = build_campaign_phase_plan(dag, phase=phase)
    submission = json.loads(submission_record_path.read_text(encoding="utf-8"))
    submission_identity = {
        key: value
        for key, value in submission.items()
        if key != "submission_record_sha256"
    }
    if submission.get("submission_record_sha256") != _stable_hash(submission_identity):
        raise ValueError("phase submission record self-hash differs")
    for field, expected in (
        ("status", "SUBMITTED"),
        ("phase", phase),
        ("run_id", dag.run_id),
        ("source_hash", dag.source_hash),
        ("config_hash", dag.config_hash),
        ("lock_hash", dag.lock_hash),
        ("campaign_inventory_hash", dag.campaign_inventory_hash),
        ("submission_plan_sha256", plan["submission_plan_sha256"]),
    ):
        if submission.get(field) != expected:
            raise ValueError(f"phase submission record {field} differs")
    expected_steps = {str(step["step_id"]) for step in plan["steps"]}
    job_ids = submission.get("submission_job_ids")
    if not isinstance(job_ids, dict) or set(job_ids) != expected_steps:
        raise ValueError("phase submission record does not exactly cover its plan")

    if evidence_dir.exists() and any(evidence_dir.iterdir()):
        raise ValueError("phase completion evidence directory must be new or empty")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    runner = subprocess.run if run_command is None else run_command
    normalized_job_ids = {str(key): str(value) for key, value in job_ids.items()}
    completed = runner(
        _sacct_array_command(
            normalized_job_ids,
            fields=("JobID", "State", "ExitCode", "ElapsedRaw", "AllocNodes"),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    raw_sacct = str(completed.stdout)
    raw_path = evidence_dir / "sacct_raw.psv"
    raw_path.write_text(raw_sacct, encoding="utf-8")
    scheduler, observed_total_au = validate_scheduler_completion_accounting(
        normalized_job_ids,
        raw_sacct,
        cpu_charge_factor=float(dag.cluster.cpu_charge_factor),
        qos_factor=float(dag.cluster.qos_factor),
        step_accounting={
            str(step["step_id"]): {
                "array_task_count": int(step["array_task_count"]),
                "shared_node_equivalent": step["shared_node_equivalent"],
            }
            for step in plan["steps"]
        },
    )

    stages_by_name = {stage.name: stage for stage in dag.stages}
    selected_stage_names = {str(step["stage"]) for step in plan["steps"]}
    reducer_evidence = {}
    for stage_name in sorted(selected_stage_names):
        stage = stages_by_name[stage_name]
        marker_path = stage.reducer_output_dir / "_SUCCESS.json"
        result_path = stage.reducer_output_dir / "reduced_result.json"
        if not marker_path.is_file() or not result_path.is_file():
            raise ValueError(f"phase stage {stage_name} lacks reducer success evidence")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("status") != "completed"
            or marker.get("stage") != stage_name
            or marker.get("output_hash") != stage.reducer_output_hash
            or marker.get("artifact_sha256") != _sha256_path(result_path)
        ):
            raise ValueError(f"phase stage {stage_name} reducer evidence differs")
        reducer_evidence[stage_name] = {
            "reducer_output_hash": stage.reducer_output_hash,
            "reduced_result_sha256": _sha256_path(result_path),
        }

    identity = {
        "schema_version": 1,
        "status": "PHASE_COMPLETE",
        "phase": phase,
        "run_id": dag.run_id,
        "source_hash": dag.source_hash,
        "config_hash": dag.config_hash,
        "lock_hash": dag.lock_hash,
        "campaign_inventory_hash": dag.campaign_inventory_hash,
        "submission_record_sha256": submission["submission_record_sha256"],
        "submission_plan_sha256": plan["submission_plan_sha256"],
        "sacct_raw_sha256": _sha256_path(raw_path),
        "observed_total_au": observed_total_au,
        "scheduler": scheduler,
        "reducers": reducer_evidence,
    }
    result = {**identity, "completion_sha256": _stable_hash(identity)}
    _write_new_json(completion_path, result)
    return result


def _cli_certificate(args: argparse.Namespace) -> int:
    summary_path = Path(args.package_summary).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    freeze = json.loads(Path(args.resource_freeze).read_text(encoding="utf-8"))
    amendment = json.loads(
        (summary_path.parent / "contract" / "fixed_family_amendment.json").read_text(
            encoding="utf-8"
        )
    )
    completion = json.loads(
        Path(args.development_completion).read_text(encoding="utf-8")
    )
    development_observed_au = _validate_development_completion(
        completion,
        expected_run_id=str(summary["run_id"]),
        resource_freeze=freeze,
    )
    allocation = freeze["allocation_accounting"]
    prior = float(allocation["prior_rejected_attempt_au"])
    accepted_pilot_observed_au = float(allocation["accepted_pilot_observed_au"])
    remaining_estimated_au = _remaining_confirmatory_estimated_au(
        summary["campaign_envelope"]["stage_allocations"]
    )
    remaining_requested_au = math.ceil(remaining_estimated_au * 1.20)
    certificate = build_campaign_budget_certificate(
        run_id=str(summary["run_id"]),
        campaign_requested_au=(
            accepted_pilot_observed_au
            + development_observed_au
            + remaining_requested_au
        ),
        prior_rejected_attempt_au=prior,
        postprocessing_reserved_au=float(args.postprocessing_reserved_au),
        allocation_quota_au=float(args.allocation_quota),
        resource_freeze_sha256=str(freeze["resource_freeze_sha256"]),
        campaign_inventory_hash=str(summary["campaign_inventory_hash"]),
        contract_amendment_sha256=str(amendment["contract_amendment_sha256"]),
        accepted_pilot_observed_au=accepted_pilot_observed_au,
        development_resolution_observed_au=development_observed_au,
        remaining_confirmatory_estimated_au=remaining_estimated_au,
        remaining_confirmatory_requested_au_with_shared_reserve=remaining_requested_au,
    )
    _write_new_json(Path(args.output), certificate)
    print(json.dumps(certificate, sort_keys=True))
    return 0


def _cli_close_pilot(args: argparse.Namespace) -> int:
    dag = load_packaged_dag(
        package_root=args.package_root,
        config_path=args.config,
        repo_root=args.repo_root,
    )
    accounting, freeze = close_completed_pilot(
        dag=dag,
        submission_record_path=Path(args.submission_record),
        accounting_output_dir=Path(args.accounting_output),
        resource_freeze_path=Path(args.resource_freeze_output),
        prior_sunk_au=float(args.prior_sunk_au),
    )
    print(
        json.dumps(
            {
                "status": "PILOT_CLOSED",
                "pilot_accounting_sha256": accounting["pilot_accounting_sha256"],
                "resource_freeze_sha256": freeze["resource_freeze_sha256"],
                "spent_through_pilot_au": freeze["allocation_accounting"][
                    "spent_through_pilot_au"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def _cli_prepare_development(args: argparse.Namespace) -> int:
    dag = prepare_development_package(
        output_dir=Path(args.output_dir),
        resource_freeze_path=Path(args.resource_freeze),
        repo_root=Path(args.repo_root),
        config_path=Path(args.config),
    )
    print(json.dumps({"status": "DEVELOPMENT_PACKAGE_READY", "run_id": dag.run_id}))
    return 0


def _cli_prepare_final(args: argparse.Namespace) -> int:
    dag, certificate = prepare_final_package(
        output_dir=Path(args.output_dir),
        resource_freeze_path=Path(args.resource_freeze),
        resolution_decision_path=Path(args.resolution_decision),
        development_completion_path=Path(args.development_completion),
        repo_root=Path(args.repo_root),
        config_path=Path(args.config),
        allocation_quota_au=float(args.allocation_quota),
        postprocessing_reserved_au=float(args.postprocessing_reserved_au),
        budget_certificate_path=Path(args.budget_certificate),
    )
    print(
        json.dumps(
            {
                "status": "FINAL_PACKAGE_READY_FOR_LIVE_PREFLIGHT",
                "run_id": dag.run_id,
                "whole_campaign_projected_au": certificate[
                    "whole_campaign_projected_au"
                ],
                "headroom_au": certificate["headroom_au"],
            },
            sort_keys=True,
        )
    )
    return 0


def _cli_adopt_generation_12(args: argparse.Namespace) -> int:
    result = write_generation_12_prerequisites(
        source_resource_freeze_path=Path(args.source_resource_freeze),
        source_resolution_decision_path=Path(args.source_resolution_decision),
        source_development_completion_path=Path(args.source_development_completion),
        repo_root=Path(args.repo_root),
        target_run_id=str(args.target_run_id),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _cli_prepare_downstream(args: argparse.Namespace) -> int:
    dag, certificate = prepare_downstream_package(
        output_dir=Path(args.output_dir),
        resource_freeze_path=Path(args.resource_freeze),
        gate_b_contract_path=Path(args.gate_b_contract),
        repo_root=Path(args.repo_root),
        config_path=Path(args.config),
        completed_observed_au=float(args.completed_observed_au),
        allocation_quota_au=float(args.allocation_quota),
        postprocessing_reserved_au=float(args.postprocessing_reserved_au),
        budget_certificate_path=Path(args.budget_certificate),
    )
    print(
        json.dumps(
            {
                "status": "DOWNSTREAM_PACKAGE_READY_FOR_LIVE_PREFLIGHT",
                "run_id": dag.run_id,
                "whole_campaign_projected_au": certificate[
                    "whole_campaign_projected_au"
                ],
                "headroom_au": certificate["headroom_au"],
            },
            sort_keys=True,
        )
    )
    return 0


def _cli_adopt_gate_b(args: argparse.Namespace) -> int:
    adoption = write_gate_b_adoption(
        gate_b_decision_path=Path(args.gate_b_decision),
        amendment_path=Path(args.amendment),
        output_path=Path(args.output),
    )
    print(json.dumps(adoption, sort_keys=True))
    return 0


def _cli_preflight_phase(args: argparse.Namespace) -> int:
    dag = load_packaged_dag(
        package_root=args.package_root,
        config_path=args.config,
        repo_root=args.repo_root,
    )
    preflight, authorization = preflight_and_authorize_phase(
        dag=dag,
        phase=args.phase,
        remaining_au=int(args.remaining_au),
        evidence_dir=Path(args.evidence_dir),
        resource_freeze_path=Path(args.resource_freeze),
        prerequisite_paths=tuple(Path(value) for value in args.prerequisite),
        authorization_path=Path(args.authorization),
    )
    print(
        json.dumps(
            {
                "status": "PHASE_AUTHORIZED",
                "phase": args.phase,
                "preflight_sha256": preflight["preflight_sha256"],
                "authorization_sha256": authorization["authorization_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def _cli_submit_phase(args: argparse.Namespace) -> int:
    if not args.execute:
        raise PermissionError("phase submission requires the explicit --execute flag")
    dag = load_packaged_dag(
        package_root=args.package_root,
        config_path=args.config,
        repo_root=args.repo_root,
    )
    job_ids = submit_authorized_phase(
        dag=dag,
        phase=args.phase,
        preflight_path=Path(args.preflight),
        authorization_path=Path(args.authorization),
        submission_record_path=Path(args.submission_record),
    )
    print(json.dumps({"status": "SUBMITTED", "phase": args.phase, "job_ids": job_ids}))
    return 0


def _cli_verify_phase(args: argparse.Namespace) -> int:
    dag = load_packaged_dag(
        package_root=args.package_root,
        config_path=args.config,
        repo_root=args.repo_root,
    )
    completion = verify_phase_completion(
        dag=dag,
        phase=args.phase,
        submission_record_path=Path(args.submission_record),
        evidence_dir=Path(args.evidence_dir),
        completion_path=Path(args.completion),
    )
    print(json.dumps(completion, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    close_pilot = subparsers.add_parser(
        "close-pilot",
        help="collect exact sacct evidence and freeze production resources",
    )
    close_pilot.add_argument("--package-root", required=True)
    close_pilot.add_argument("--config", required=True)
    close_pilot.add_argument("--repo-root", required=True)
    close_pilot.add_argument("--submission-record", required=True)
    close_pilot.add_argument("--accounting-output", required=True)
    close_pilot.add_argument("--resource-freeze-output", required=True)
    close_pilot.add_argument("--prior-sunk-au", required=True, type=float)
    close_pilot.set_defaults(func=_cli_close_pilot)

    development = subparsers.add_parser(
        "prepare-development",
        help="generate the post-pilot adaptive-resolution package",
    )
    development.add_argument("--output-dir", required=True)
    development.add_argument("--resource-freeze", required=True)
    development.add_argument("--repo-root", required=True)
    development.add_argument("--config", required=True)
    development.set_defaults(func=_cli_prepare_development)

    final = subparsers.add_parser(
        "prepare-final",
        help="generate and allocation-certify the final confirmatory package",
    )
    final.add_argument("--output-dir", required=True)
    final.add_argument("--resource-freeze", required=True)
    final.add_argument("--resolution-decision", required=True)
    final.add_argument("--development-completion", required=True)
    final.add_argument("--repo-root", required=True)
    final.add_argument("--config", required=True)
    final.add_argument("--allocation-quota", type=float, default=25_000)
    final.add_argument("--postprocessing-reserved-au", type=float, default=5.0)
    final.add_argument("--budget-certificate", required=True)
    final.set_defaults(func=_cli_prepare_final)

    generation_12 = subparsers.add_parser(
        "adopt-generation-12-prerequisites",
        help=(
            "bind invariant pilot telemetry and the accepted resolution to a fresh "
            "generation-12 package"
        ),
    )
    generation_12.add_argument("--source-resource-freeze", required=True)
    generation_12.add_argument("--source-resolution-decision", required=True)
    generation_12.add_argument("--source-development-completion", required=True)
    generation_12.add_argument("--repo-root", required=True)
    generation_12.add_argument("--target-run-id", required=True)
    generation_12.add_argument("--output-dir", required=True)
    generation_12.set_defaults(func=_cli_adopt_generation_12)

    downstream = subparsers.add_parser(
        "prepare-downstream",
        help="generate fixed-family and later stages without repeating Gate B",
    )
    downstream.add_argument("--output-dir", required=True)
    downstream.add_argument("--resource-freeze", required=True)
    downstream.add_argument("--gate-b-contract", required=True)
    downstream.add_argument("--repo-root", required=True)
    downstream.add_argument("--config", required=True)
    downstream.add_argument("--completed-observed-au", required=True, type=float)
    downstream.add_argument("--allocation-quota", type=float, default=30_000)
    downstream.add_argument("--postprocessing-reserved-au", type=float, default=5.0)
    downstream.add_argument("--budget-certificate", required=True)
    downstream.set_defaults(func=_cli_prepare_downstream)

    adoption = subparsers.add_parser(
        "adopt-gate-b",
        help="bind an exact Gate-B PASS to the fixed-family-only amendment",
    )
    adoption.add_argument("--gate-b-decision", required=True)
    adoption.add_argument("--amendment", required=True)
    adoption.add_argument("--output", required=True)
    adoption.set_defaults(func=_cli_adopt_gate_b)

    preflight = subparsers.add_parser(
        "preflight-phase", help="run live probes and write one phase authorization"
    )
    preflight.add_argument("--package-root", required=True)
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--repo-root", required=True)
    preflight.add_argument(
        "--phase",
        required=True,
        choices=("development", "gate_b", "fixed_family", "gate_p", "gate_c"),
    )
    preflight.add_argument("--remaining-au", required=True, type=int)
    preflight.add_argument("--evidence-dir", required=True)
    preflight.add_argument("--resource-freeze", required=True)
    preflight.add_argument("--prerequisite", action="append", default=[])
    preflight.add_argument("--authorization", required=True)
    preflight.set_defaults(func=_cli_preflight_phase)

    submit = subparsers.add_parser(
        "submit-phase",
        help="submit exactly one already-preflighted and authorized phase",
    )
    submit.add_argument("--package-root", required=True)
    submit.add_argument("--config", required=True)
    submit.add_argument("--repo-root", required=True)
    submit.add_argument(
        "--phase",
        required=True,
        choices=("development", "gate_b", "fixed_family", "gate_p", "gate_c"),
    )
    submit.add_argument("--preflight", required=True)
    submit.add_argument("--authorization", required=True)
    submit.add_argument("--submission-record", required=True)
    submit.add_argument("--execute", action="store_true")
    submit.set_defaults(func=_cli_submit_phase)

    verify = subparsers.add_parser(
        "verify-phase",
        help="require exact scheduler and reducer completion for one phase",
    )
    verify.add_argument("--package-root", required=True)
    verify.add_argument("--config", required=True)
    verify.add_argument("--repo-root", required=True)
    verify.add_argument(
        "--phase",
        required=True,
        choices=("development", "gate_b", "fixed_family", "gate_p", "gate_c"),
    )
    verify.add_argument("--submission-record", required=True)
    verify.add_argument("--evidence-dir", required=True)
    verify.add_argument("--completion", required=True)
    verify.set_defaults(func=_cli_verify_phase)

    certificate = subparsers.add_parser(
        "certify-budget",
        help="bind rejected-attempt AUs to a final package's conservative envelope",
    )
    certificate.add_argument("--package-summary", required=True)
    certificate.add_argument("--resource-freeze", required=True)
    certificate.add_argument("--development-completion", required=True)
    certificate.add_argument("--allocation-quota", required=True, type=float)
    certificate.add_argument("--postprocessing-reserved-au", type=float, default=5.0)
    certificate.add_argument("--output", required=True)
    certificate.set_defaults(func=_cli_certificate)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
