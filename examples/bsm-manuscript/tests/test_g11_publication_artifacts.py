"""Publication artifact compiler tests using production-scale output accounting."""

from __future__ import annotations

import json
import hashlib
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_g11_publication_artifacts import build_publication_artifacts
from scripts.audit_g11_publication_artifacts import audit_publication_artifacts
from scripts.install_g11_manuscript_artifacts import install_manuscript_artifacts
from scripts.finalize_g11_manuscript_release import finalize_manuscript_release
from scripts.reproduce_artifacts import MANUSCRIPT_FIGURES


MODELS = (
    "null_mean",
    "main_effects_ols",
    "screened_ols",
    "penalized_ols",
    "final_ols",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_bound_terminal(path: Path, payload: dict[str, object]) -> None:
    _write_json(path, payload)
    _write_json(
        path.parent / "result.json",
        {
            "operation": payload["operation"],
            "status": "completed",
            "terminal_record": str(path),
            "scientific_artifacts": [
                {
                    "path": str(path),
                    "relative_path": "terminal_record.json",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }
            ],
        },
    )


def _write_stage_summary(results_root: Path, stage: str, **values: object) -> None:
    _write_json(
        results_root / "stages" / stage / "reducer" / "scientific_stage_summary.json",
        {"operation": stage, "status": "completed", **values},
    )


def _write_reducer_proofs(
    package_root: Path,
    results_root: Path,
    *,
    contract_hash: str,
    stages: tuple[str, ...],
) -> None:
    package = json.loads(
        (package_root / "package_summary.json").read_text(encoding="utf-8")
    )
    provenance_source = results_root.parent / "provenance_source"
    data_root = provenance_source / "applied_data"
    data_root.mkdir(parents=True, exist_ok=True)
    dataset_manifest = data_root / "dataset_manifest.json"
    feature_catalog = data_root / "metadata" / "manuscript_feature_catalog.parquet"
    feature_catalog.parent.mkdir()
    pd.DataFrame(
        {
            "feature_name": [
                *[f"continuous_{index:03d}" for index in range(158)],
                "FM.Use Agnostic FS Conversion",
                "OI.Use AEO Reference Oil",
            ],
            "feature_type": ["first_order"] * 160,
        }
    ).to_parquet(feature_catalog, index=False)
    dataset_identity = {
        "schema_version": 1,
        "status": "PREPARED_HOLDOUT_SEALED",
        "dimensions": {
            "rows": 30000,
            "train_rows": 28500,
            "holdout_rows": 1500,
            "inputs": 160,
            "continuous_inputs": 158,
            "binary_inputs": 2,
            "outputs": 23495,
        },
        "binary_input_names": [
            "FM.Use Agnostic FS Conversion",
            "OI.Use AEO Reference Oil",
        ],
        "generated_sha256": {
            "metadata/manuscript_feature_catalog.parquet": hashlib.sha256(
                feature_catalog.read_bytes()
            ).hexdigest()
        },
    }
    dataset_identity["dataset_manifest_sha256"] = hashlib.sha256(
        json.dumps(dataset_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json(dataset_manifest, dataset_identity)
    applied_config = provenance_source / "manuscript_case_study.yml"
    applied_config.write_text("case_study: {}\n", encoding="utf-8")
    adapter = provenance_source / "g11_campaign_adapter.py"
    adapter.write_text("# fixture adapter\n", encoding="utf-8")
    package_stages = []
    for stage in stages:
        reducer = results_root / "stages" / stage / "reducer"
        reducer.mkdir(parents=True, exist_ok=True)
        output_hash = hashlib.sha256(f"reducer:{stage}".encode()).hexdigest()
        scientific_artifacts = [
            {
                "path": str(path.resolve()),
                "relative_path": path.relative_to(reducer).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in sorted(reducer.rglob("*"))
            if path.is_file()
        ]
        worker_results = sorted(
            (results_root / "stages" / stage / "results").glob("*/result.json")
        )
        artifact_hashes = (
            [hashlib.sha256(path.read_bytes()).hexdigest() for path in worker_results]
            if worker_results
            else [hashlib.sha256(f"worker:{stage}".encode()).hexdigest()]
        )
        reduced = {
            "schema_version": 2,
            "stage": stage,
            "records": 1,
            "contract_hash": contract_hash,
            "parent_hash": hashlib.sha256(f"parent:{stage}".encode()).hexdigest(),
            "artifact_hashes": artifact_hashes,
            "scientific_reduction": {
                "operation": stage,
                "status": "completed",
                "scientific_artifacts": scientific_artifacts,
            },
            "status": "completed",
        }
        reduced_path = reducer / "reduced_result.json"
        _write_json(reduced_path, reduced)
        _write_json(
            reducer / "_SUCCESS.json",
            {
                "schema_version": 2,
                "stage": stage,
                "status": "completed",
                "output_hash": output_hash,
                "artifact_sha256": hashlib.sha256(
                    reduced_path.read_bytes()
                ).hexdigest(),
            },
        )
        manifest_path = package_root / "stages" / stage / "manifest.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "stage": stage,
            "applied_data_root": str(data_root),
            "applied_data_manifest_sha256": hashlib.sha256(
                dataset_manifest.read_bytes()
            ).hexdigest(),
            "applied_config_path": str(applied_config),
            "applied_config_sha256": hashlib.sha256(
                applied_config.read_bytes()
            ).hexdigest(),
            "scientific_adapter_path": str(adapter),
            "scientific_adapter_sha256": hashlib.sha256(
                adapter.read_bytes()
            ).hexdigest(),
        }
        manifest_path.write_text(
            json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
        )
        package_stages.append(
            {
                "name": stage,
                "partition": "shared",
                "job_count": 1,
                "reducer_output_hash": output_hash,
                "manifest_path": str(manifest_path),
            }
        )
    package["stages"] = package_stages
    _write_json(package_root / "package_summary.json", package)


def _build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    package_root = tmp_path / "package"
    results_root = tmp_path / "results"
    run_id = "g11-publication-final"
    contract_path = package_root / "contract" / "g11_campaign_contract.toml"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        Path(__file__).resolve().parents[1] / "configs" / "g11_campaign_contract.toml",
        contract_path,
    )
    from rfm_pipeline.campaign_contract import G11_CONTRACT, load_contract

    contract, contract_hash = load_contract(contract_path)
    from scripts.g11_campaign_workflow import build_publication_contract_amendment

    _, provisional_amendment = build_publication_contract_amendment(G11_CONTRACT)
    amendment_identity = {
        key: value
        for key, value in provisional_amendment.items()
        if key != "contract_amendment_sha256"
    }
    amendment_identity["selected_contract_hash"] = contract_hash
    amendment = {
        **amendment_identity,
        "contract_amendment_sha256": hashlib.sha256(
            json.dumps(
                amendment_identity, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }
    assert contract.fixed_family_replicates == 300
    _write_json(package_root / "contract" / "fixed_family_amendment.json", amendment)
    source_hash = "b" * 64
    lock_hash = "c" * 64
    inventory_path = package_root / "campaign_inventory.jsonl"
    inventory_path.write_text('{"path":"fixture"}\n', encoding="utf-8")
    inventory_hash = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    _write_json(
        package_root / "package_summary.json",
        {
            "run_id": run_id,
            "package_mode": "confirmatory",
            "source_hash": source_hash,
            "config_hash": contract_hash,
            "lock_hash": lock_hash,
            "campaign_inventory_path": str(inventory_path),
            "campaign_inventory_hash": inventory_hash,
            "campaign_envelope": {"requested_au": 1234},
        },
    )

    _write_stage_summary(
        results_root,
        "applied_conditioning",
        retained_output_count=9954,
        culled_output_count=13541,
    )
    conditioning_artifacts = (
        results_root
        / "stages"
        / "applied_conditioning"
        / "reducer"
        / "output_conditioning"
    )
    conditioning_artifacts.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"n_components_retained": 17}]).to_csv(
        conditioning_artifacts / "output_conditioning_summary.csv", index=False
    )
    _write_stage_summary(results_root, "applied_screening", retained_term_count=72)
    _write_stage_summary(
        results_root,
        "applied_interaction",
        candidate_pair_count=2556,
        retained_pair_count=147,
        draw_count=999,
    )
    _write_stage_summary(
        results_root,
        "applied_nonlinear",
        retained_transformation_count=23,
    )
    _write_stage_summary(
        results_root,
        "applied_sparse_resample",
        final_stable_support_count=342,
        resample_count=50,
    )
    _write_stage_summary(
        results_root,
        "applied_terminal_fit",
        final_count=3,
    )

    output_names = [f"output_{index:05d}" for index in range(23495)]
    feature_names = ["CHC.x", "OHC.y:WW.z", "AHC.q_sq"]
    terminal = results_root / "stages" / "applied_terminal_fit" / "reducer" / "terminal"
    from rfm_pipeline.manuscript_stages import (
        FrozenPredictionMatrices,
        _compute_frozen_model_digest,
        _compute_model_freeze_hash,
        _g11_eligibility_ledger_from_statistics,
        write_frozen_prediction_matrices,
    )

    (terminal / "model").mkdir(parents=True, exist_ok=True)
    coefficients = np.ones((3, 23495), dtype=np.float64)
    intercept = np.zeros(23495, dtype=np.float64)
    x_means = np.asarray([1.0, 2.0, 3.0])
    x_scales = np.asarray([2.0, 4.0, 5.0])
    y_min = np.zeros(23495, dtype=np.float64)
    y_max = np.ones(23495, dtype=np.float64)
    y_mean = np.zeros(23495, dtype=np.float64)
    y_variance = np.r_[np.ones(9954), np.zeros(13541)]
    model_digest = _compute_frozen_model_digest(coefficients, intercept)
    freeze_hash = _compute_model_freeze_hash(
        contract_hash,
        tuple(feature_names),
        tuple(output_names),
        10,
        tuple(y_min),
        tuple(y_max),
        tuple(y_mean),
        tuple(y_variance),
        tuple(x_means),
        tuple(x_scales),
        model_digest,
    )
    np.savez_compressed(
        terminal / "model" / "frozen_model.npz",
        coef=coefficients,
        intercept=intercept,
        x_means=x_means,
        x_scales=x_scales,
    )
    model_path = terminal / "model" / "frozen_model.npz"
    _write_json(
        terminal / "model" / "freeze_manifest.json",
        {
            "schema_version": 1,
            "freeze_hash": freeze_hash,
            "contract_hash": contract_hash,
            "feature_names": feature_names,
            "output_names": output_names,
            "n_train_rows": 10,
            "y_train_min": y_min.tolist(),
            "y_train_max": y_max.tolist(),
            "y_train_mean": y_mean.tolist(),
            "y_train_variance": y_variance.tolist(),
            "x_means": x_means.tolist(),
            "x_scales": x_scales.tolist(),
            "model_digest": model_digest,
            "model_npz_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        },
    )
    pd.DataFrame(
        {"feature_name": feature_names, "hc3_retained_after_filter": [True] * 3}
    ).to_csv(terminal / "hc3_inferential_filter_summary.csv", index=False)
    pd.DataFrame(
        {
            "feature_name": feature_names,
            "remove_rank": [1, 2, 3],
            "retained_features": [2, 1, 0],
            "approx_macro_nrmse_upper_bound": [0.11, 0.12, 0.13],
            "selected_by_effective_cutoff": [False] * 3,
        }
    ).to_csv(terminal / "feature_pruning_impact.csv", index=False)
    pd.DataFrame(
        [
            {
                "final_refit_n_features": 3,
                "auto_remove_count": 0,
                "effective_remove_count": 0,
            }
        ]
    ).to_csv(terminal / "feature_pruning_summary.csv", index=False)
    table_names = (
        "hc3_inferential_filter_summary.csv",
        "feature_pruning_impact.csv",
        "feature_pruning_summary.csv",
    )
    _write_json(
        terminal / "terminal_manifest.json",
        {
            "schema_version": 1,
            "freeze_hash": freeze_hash,
            "prefilter_feature_names": [*feature_names, "SE.removed"],
            "hc3_feature_names": feature_names,
            "final_feature_names": feature_names,
            "model_manifest_sha256": hashlib.sha256(
                (terminal / "model" / "freeze_manifest.json").read_bytes()
            ).hexdigest(),
            "model_npz_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "table_sha256": {
                name: hashlib.sha256((terminal / name).read_bytes()).hexdigest()
                for name in table_names
            },
        },
    )
    terminal_summary_path = (
        results_root
        / "stages"
        / "applied_terminal_fit"
        / "reducer"
        / "scientific_stage_summary.json"
    )
    terminal_summary = json.loads(terminal_summary_path.read_text(encoding="utf-8"))
    terminal_summary["freeze_hash"] = freeze_hash
    _write_json(terminal_summary_path, terminal_summary)

    ablation = results_root / "stages" / "applied_ablation_fit" / "reducer" / "models"
    feature_counts = [0, 160, 72, 342, 3]
    for model, count in zip(MODELS, feature_counts, strict=True):
        _write_json(
            ablation / model / "freeze_manifest.json",
            {"feature_names": [f"f{i}" for i in range(count)]},
        )

    bootstrap = results_root / "stages" / "applied_bootstrap" / "reducer"
    model_summaries = {}
    for index, model in enumerate(MODELS):
        summary = {
            "model": model,
            "metric": "macro_nrmse",
            "point_estimate": 0.16 - index * 0.02,
            "ci_lower": 0.15 - index * 0.02,
            "ci_upper": 0.17 - index * 0.02,
            "draw_count": 2000,
            "eligible_count": 9954,
            "freeze_hash": freeze_hash if model == "final_ols" else str(index) * 64,
            "schedule_hash": "9" * 64,
        }
        _write_json(bootstrap / "models" / f"{model}.json", summary)
        model_summaries[model] = summary
    _write_stage_summary(
        results_root,
        "applied_bootstrap",
        contract_hash=contract_hash,
        draw_count=2000,
        eligible_count=9954,
        models=model_summaries,
        paired_contrasts={},
    )

    ledger = _g11_eligibility_ledger_from_statistics(
        output_ids=tuple(output_names),
        ref_min=y_min,
        ref_max=y_max,
        ref_mean=y_mean,
        ref_variance=y_variance,
    )
    eligibility = results_root / "stages" / "applied_eligibility" / "reducer"
    eligibility.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(eligibility / "output_eligibility_ledger.csv", index=False)

    predictions = (
        results_root
        / "stages"
        / "applied_holdout_predict"
        / "reducer"
        / "models"
        / "final_ols"
    )
    truth = np.zeros((2, 23495), dtype=np.float64)
    prediction = np.zeros_like(truth)
    prediction[:, :9954] = 0.2
    write_frozen_prediction_matrices(
        frozen=FrozenPredictionMatrices(
            freeze_hash=freeze_hash,
            y_holdout=truth,
            y_pred=prediction,
            y_train_min=y_min,
            y_train_max=y_max,
            y_train_mean=y_mean,
            y_train_variance=y_variance,
            output_names=tuple(output_names),
            truth_ids=("holdout-1", "holdout-2"),
            prediction_ids=("holdout-1", "holdout-2"),
            strata=("00", "11"),
            eligibility_ledger=ledger,
        ),
        output_dir=predictions,
    )

    _write_json(
        results_root / "stages" / "gate_b" / "reducer" / "gate_b_decision.json",
        {
            "operation": "gate_b",
            "status": "completed",
            "decision": "PASS",
            "contract_hash": contract_hash,
            "terminal_record_count": 1,
            "scenarios": {
                "global_null": {
                    "denominator": 100,
                    "false_selection_events": 1,
                    "fwer": 0.01,
                    "one_sided_wilson_upper": 0.040,
                    "passes_calibration": True,
                }
            },
        },
    )
    _write_json(
        results_root
        / "stages"
        / "fixed_family_supplement"
        / "reducer"
        / "fixed_family_decision.json",
        {
            "operation": "fixed_family_supplement",
            "status": "completed",
            "decision": "PASS",
            "contract_hash": contract_hash,
            "terminal_record_count": 1,
            "families": {},
        },
    )
    _write_json(
        results_root
        / "stages"
        / "recovery"
        / "reducer"
        / "gate_c_stress_ledger_summary.json",
        {
            "operation": "recovery",
            "status": "completed",
            "acceptance_status": "READY_FOR_INDEPENDENT_REVIEW",
            "contract_hash": contract_hash,
            "terminal_record_count": 1,
        },
    )
    _write_bound_terminal(
        results_root
        / "stages"
        / "gate_b"
        / "results"
        / "task-0000"
        / "terminal_record.json",
        {
            "operation": "gate_b",
            "scenario": "global_null",
            "replicate_index": 0,
            "seed": 100,
            "contract_hash": contract_hash,
            "status": "completed",
            "recovery_metrics": {},
            "macro_nrmse_by_method": {},
        },
    )
    _write_bound_terminal(
        results_root
        / "stages"
        / "fixed_family_supplement"
        / "results"
        / "task-0000"
        / "terminal_record.json",
        {
            "operation": "fixed_family_supplement",
            "global_null_replicate_index": 0,
            "family_sizes": [],
            "selected_pairs": {},
            "status": "completed",
        },
    )
    terminal_path = (
        results_root
        / "stages"
        / "recovery"
        / "results"
        / "task-0000"
        / "terminal_record.json"
    )
    _write_bound_terminal(
        terminal_path,
        {
            "operation": "recovery",
            "scenario": "stress",
            "replicate_index": 0,
            "seed": 123,
            "contract_hash": contract_hash,
            "status": "completed",
            "truth_interaction_ids": ["HEFA:HTL"],
            "retained_interaction_ids": ["HTL:HEFA"],
            "in_library_truth_support": ["main_effect", "HEFA:HTL"],
            "final_selected_support": ["main_effect", "HTL:HEFA"],
            "recovery_metrics": {
                "whole": {
                    "precision": 0.5,
                    "recall": 0.5,
                    "exact_support_recovery": False,
                },
                "interaction": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "exact_support_recovery": False,
                },
            },
            "macro_nrmse_by_method": {"proposed_terminal_workflow": 0.04},
        },
    )
    required_stages = (
        "applied_conditioning",
        "applied_screening",
        "applied_interaction",
        "applied_nonlinear",
        "applied_sparse_resample",
        "applied_terminal_fit",
        "applied_ablation_fit",
        "applied_holdout_predict",
        "applied_eligibility",
        "applied_bootstrap",
        "gate_b",
        "fixed_family_supplement",
        "recovery",
    )
    _write_reducer_proofs(
        package_root,
        results_root,
        contract_hash=contract_hash,
        stages=required_stages,
    )
    return package_root, results_root


def test_compiler_emits_complete_machine_readable_and_latex_surfaces(
    tmp_path: Path,
) -> None:
    package_root, results_root = _build_fixture(tmp_path)
    output_root = tmp_path / "publication"

    manifest = build_publication_artifacts(
        package_root=package_root,
        results_root=results_root,
        output_root=output_root,
    )

    assert manifest["status"] == "PUBLICATION_ARTIFACTS_COMPLETE"
    assert (output_root / "tables" / "output_eligibility_ledger.csv").is_file()
    assert (output_root / "tables" / "per_output_nrmse.csv").is_file()
    assert (output_root / "tables" / "ablation_table.csv").is_file()
    assert (output_root / "tables" / "recovery_replicates.csv").is_file()
    assert (output_root / "tables" / "recovery_scenario_summary.csv").is_file()
    recovery_rows = pd.read_csv(output_root / "tables" / "recovery_replicates.csv")
    recovery = json.loads(
        recovery_rows.loc[
            recovery_rows["scenario"] == "stress", "recovery_metrics_json"
        ].item()
    )
    assert recovery["interaction"]["recall"] == pytest.approx(1.0)
    assert recovery["interaction"]["false_discovery_proportion"] == pytest.approx(
        0.0
    )
    assert recovery["whole"]["exact_support_recovery"] is True
    input_catalog = pd.read_csv(output_root / "metadata" / "input_feature_catalog.csv")
    assert len(input_catalog) == 160
    assert {
        "FM.Use Agnostic FS Conversion",
        "OI.Use AEO Reference Oil",
    } <= set(input_catalog["feature_name"])
    assert (
        output_root / "figure_data" / "figure_support_composition_data.csv"
    ).is_file()
    assert (output_root / "final_model" / "coefficient_matrix_raw_scale.csv").is_file()
    assert (output_root / "manuscript" / "manuscript_results.tex").is_file()
    assert (output_root / "manuscript" / "recovery_fwer_rows.tex").is_file()
    assert (output_root / "manuscript" / "recovery_scenario_rows.tex").is_file()
    public_summary = json.loads(
        (output_root / "provenance" / "campaign_package_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert "campaign_inventory_path" not in public_summary
    assert all("manifest_path" not in stage for stage in public_summary["stages"])
    assert not (output_root / "provenance" / "campaign_inventory.jsonl").exists()
    copied_amendment = json.loads(
        (output_root / "provenance" / "fixed_family_amendment.json").read_text(
            encoding="utf-8"
        )
    )
    assert copied_amendment["changed_fields"] == {
        "fixed_family_replicates": {"before": 1000, "after": 300}
    }
    tex = (output_root / "manuscript" / "manuscript_results.tex").read_text(
        encoding="utf-8"
    )
    assert r"\newcommand{\FinalPredictorCount}{3}" in tex
    assert r"\newcommand{\EligibleOutputCount}{9,954}" in tex
    assert len(manifest["artifacts"]) >= 15

    observed = pd.read_csv(output_root / "tables" / "per_output_nrmse.csv")
    assert len(observed) == 23495
    assert int(observed["included_in_macro"].sum()) == 9954

    from rfm_pipeline import regenerate_figures_from_committed_data

    written = regenerate_figures_from_committed_data(
        figure_data_dir=output_root / "figure_data",
        tables_dir=output_root / "tables",
        output_dir=output_root / "figures",
    )
    assert len(written) == 11
    assert all(path.stat().st_size > 1024 for path in written.values())


def test_compiler_rejects_results_not_bound_to_the_confirmatory_package(
    tmp_path: Path,
) -> None:
    package_root, results_root = _build_fixture(tmp_path)
    marker_path = (
        results_root / "stages" / "applied_bootstrap" / "reducer" / "_SUCCESS.json"
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["output_hash"] = "f" * 64
    _write_json(marker_path, marker)

    with pytest.raises(ValueError, match="reducer output identity"):
        build_publication_artifacts(
            package_root=package_root,
            results_root=results_root,
            output_root=tmp_path / "publication",
        )


def test_compiler_rejects_terminal_recovery_bytes_changed_after_worker_result(
    tmp_path: Path,
) -> None:
    package_root, results_root = _build_fixture(tmp_path)
    terminal = (
        results_root
        / "stages"
        / "recovery"
        / "results"
        / "task-0000"
        / "terminal_record.json"
    )
    payload = json.loads(terminal.read_text(encoding="utf-8"))
    payload["seed"] = 999
    _write_json(terminal, payload)

    with pytest.raises(ValueError, match="not bound to its worker result"):
        build_publication_artifacts(
            package_root=package_root,
            results_root=results_root,
            output_root=tmp_path / "publication",
        )


def test_independent_audit_and_manuscript_install_are_identity_bound(
    tmp_path: Path,
) -> None:
    package_root, results_root = _build_fixture(tmp_path)
    publication = tmp_path / "publication"
    build_publication_artifacts(
        package_root=package_root,
        results_root=results_root,
        output_root=publication,
    )
    figures = tmp_path / "figures"
    figures.mkdir()
    pdf = b"%PDF-1.4\n" + (b"0" * 1100) + b"\nstartxref\n0\n%%EOF\n"
    for name in MANUSCRIPT_FIGURES:
        (figures / name).write_bytes(pdf)
    audit_path = tmp_path / "publication_audit.json"
    audit = audit_publication_artifacts(
        publication_root=publication,
        figures_root=figures,
        output_path=audit_path,
    )
    assert audit["status"] == "PUBLICATION_ARTIFACT_AUDIT_PASS"
    assert "publication_root" not in audit
    assert "figures_root" not in audit

    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()
    (manuscript / "manuscript.tex").write_text("manuscript", encoding="utf-8")
    installed = install_manuscript_artifacts(
        publication_root=publication,
        figures_root=figures,
        audit_path=audit_path,
        manuscript_root=manuscript,
    )
    assert installed["status"] == "MANUSCRIPT_ARTIFACTS_INSTALLED"
    assert (manuscript / "generated" / "manuscript_results.tex").is_file()
    assert (manuscript / "generated" / "recovery_fwer_rows.tex").is_file()
    assert (manuscript / "generated" / "recovery_scenario_rows.tex").is_file()
    assert all((manuscript / "figures" / name).is_file() for name in MANUSCRIPT_FIGURES)


def test_independent_audit_recomputes_raw_scale_model_algebra(
    tmp_path: Path,
) -> None:
    from scripts.g11_campaign_workflow import _stable_hash

    package_root, results_root = _build_fixture(tmp_path)
    publication = tmp_path / "publication"
    build_publication_artifacts(
        package_root=package_root,
        results_root=results_root,
        output_root=publication,
    )
    raw_path = publication / "final_model" / "coefficient_matrix_raw_scale.csv"
    rows = raw_path.read_text(encoding="utf-8").splitlines()
    first = rows[1].split(",")
    first[1] = str(float(first[1]) + 1.0)
    rows[1] = ",".join(first)
    raw_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    manifest_path = publication / "publication_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        if item["relative_path"] == "final_model/coefficient_matrix_raw_scale.csv":
            item["sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            item["bytes"] = raw_path.stat().st_size
    manifest["artifact_set_sha256"] = _stable_hash(manifest["artifacts"])
    _write_json(manifest_path, manifest)

    figures = tmp_path / "figures"
    figures.mkdir()
    pdf = b"%PDF-1.4\n" + (b"0" * 1100) + b"\nstartxref\n0\n%%EOF\n"
    for name in MANUSCRIPT_FIGURES:
        (figures / name).write_bytes(pdf)

    with pytest.raises(ValueError, match="raw-scale coefficient values differ"):
        audit_publication_artifacts(
            publication_root=publication,
            figures_root=figures,
        )


def test_independent_audit_rejects_rehashed_fixed_family_contract_drift(
    tmp_path: Path,
) -> None:
    from scripts.g11_campaign_workflow import _stable_hash

    package_root, results_root = _build_fixture(tmp_path)
    publication = tmp_path / "publication"
    build_publication_artifacts(
        package_root=package_root,
        results_root=results_root,
        output_root=publication,
    )
    amendment_path = publication / "provenance" / "fixed_family_amendment.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    amendment["changed_fields"]["fixed_family_replicates"]["after"] = 201
    amendment_identity = {
        key: value
        for key, value in amendment.items()
        if key != "contract_amendment_sha256"
    }
    amendment["contract_amendment_sha256"] = _stable_hash(amendment_identity)
    _write_json(amendment_path, amendment)
    manifest_path = publication / "publication_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        if item["relative_path"] == "provenance/fixed_family_amendment.json":
            item["sha256"] = hashlib.sha256(amendment_path.read_bytes()).hexdigest()
            item["bytes"] = amendment_path.stat().st_size
    manifest["artifact_set_sha256"] = _stable_hash(manifest["artifacts"])
    _write_json(manifest_path, manifest)
    figures = tmp_path / "figures"
    figures.mkdir()
    pdf = b"%PDF-1.4\n" + (b"0" * 1100) + b"\nstartxref\n0\n%%EOF\n"
    for name in MANUSCRIPT_FIGURES:
        (figures / name).write_bytes(pdf)

    with pytest.raises(ValueError, match="fixed-family publication amendment"):
        audit_publication_artifacts(
            publication_root=publication,
            figures_root=figures,
        )


def test_local_finalizer_builds_audited_release_without_scientific_compute(
    tmp_path: Path,
) -> None:
    package_root, results_root = _build_fixture(tmp_path)
    publication = tmp_path / "publication"
    build_publication_artifacts(
        package_root=package_root,
        results_root=results_root,
        output_root=publication,
    )
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()
    (manuscript / "manuscript.tex").write_text("manuscript", encoding="utf-8")
    (manuscript / "coverpage.tex").write_text("cover", encoding="utf-8")
    for name in (
        "ref.bib",
        "acmart.cls",
        "ACM-Reference-Format.bst",
        "acm-ims-jds-logo.pdf",
        "acm-jdslogo.png",
    ):
        (manuscript / name).write_bytes((name + "\n").encode("utf-8"))
    bsm_repo = Path(__file__).resolve().parents[1]
    release = tmp_path / "release"

    def fake_run(command: list[str], **_kwargs: object) -> object:
        import subprocess

        if "reproduce_artifacts.py" in " ".join(command):
            figures = Path(command[command.index("--output-dir") + 1])
            figures.mkdir(parents=True, exist_ok=True)
            for index in range(11):
                (figures / f"figure-{index}.svg").write_text(
                    "<svg>" + ("x" * 1100) + "</svg>", encoding="utf-8"
                )
            pdf = b"%PDF-1.4\n" + (b"0" * 1100) + b"\nstartxref\n0\n%%EOF\n"
            for name in MANUSCRIPT_FIGURES:
                (figures / name).write_bytes(pdf)
        elif command[0] == "latexmk":
            output = Path(
                next(
                    value.split("=", 1)[1]
                    for value in command
                    if value.startswith("-outdir=")
                )
            )
            output.mkdir(parents=True, exist_ok=True)
            (output / Path(command[-1]).with_suffix(".pdf").name).write_bytes(
                b"%PDF-1.4\n" + (b"0" * 1100) + b"\nstartxref\n0\n%%EOF\n"
            )
            (output / Path(command[-1]).with_suffix(".log").name).write_text(
                "Output written successfully.\n", encoding="utf-8"
            )
        elif command[0] == "pdfinfo":
            pages = 1 if command[1].endswith("coverpage.pdf") else 21
            return subprocess.CompletedProcess(
                command, 0, stdout=f"Pages:           {pages}\n", stderr=""
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = finalize_manuscript_release(
        publication_root=publication,
        bsm_repo_root=bsm_repo,
        manuscript_root=manuscript,
        release_root=release,
        replace=True,
        run_command=fake_run,
    )

    assert result["status"] == "JDS_RELEASE_BUILD_PASS"
    assert len(list((release / "figures").glob("*.svg"))) == 11
    assert (release / "pdf" / "manuscript.pdf").is_file()
    assert (release / "pdf" / "coverpage.pdf").is_file()
    assert result["manuscript_page_count"] == 21
    assert result["coverpage_page_count"] == 1
    source_zip = release / "jds-submission-source.zip"
    supplement_zip = release / "bsm-public-rf-supplement.zip"
    assert source_zip.is_file()
    assert supplement_zip.is_file()
    with zipfile.ZipFile(source_zip) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
    assert "manuscript.tex" in names
    assert "coverpage.tex" in names
    assert "generated/manuscript_results.tex" in names
    assert "figures/figure_nrmse_bootstrap_summary.pdf" in names
    with zipfile.ZipFile(supplement_zip) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
    assert "publication_artifact_manifest.json" in names
    assert "publication_artifact_audit.json" in names
    assert "README.md" in names
    assert "LICENSE" in names
    assert "reproduction/scripts/reproduce_artifacts.py" in names
    assert "reproduction/pixi.toml" in names
    assert "reproduction/pixi.lock" in names
    with zipfile.ZipFile(supplement_zip) as archive:
        supplement_readme = archive.read("README.md").decode("utf-8")
    assert "PUBLICATION_ARTIFACT_AUDIT_PASS" in supplement_readme
    assert "pixi install --locked" in supplement_readme
    assert "--artifact-root .." in supplement_readme
    assert result["submission_archive_sha256"]["jds-submission-source.zip"]
    assert result["submission_archive_sha256"]["bsm-public-rf-supplement.zip"]
