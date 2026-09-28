#!/usr/bin/env python
"""Content-addressed BSM adapter for one G11 calibration/recovery work unit."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
_DRIVER_PATH = ROOT / "scripts" / "run_bsm_recovery_study.py"


def _require_hashed_file(
    record: dict[str, Any], *, path_field: str, hash_field: str
) -> Path:
    raw_path = record.get(path_field)
    if not raw_path:
        raise ValueError(f"manifest record is missing {path_field}")
    path = Path(str(raw_path))
    if not path.is_file():
        raise ValueError(f"manifest-bound file does not exist: {path_field}={path}")
    expected = str(record.get(hash_field, ""))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"manifest-bound {path_field} hash differs")
    return path


def _load_driver(record: dict[str, Any]) -> ModuleType:
    driver_path = _require_hashed_file(
        record,
        path_field="bsm_recovery_driver_path",
        hash_field="bsm_recovery_driver_sha256",
    )
    if driver_path.resolve() != _DRIVER_PATH.resolve():
        raise ValueError("manifest-bound recovery driver path is not canonical")
    module_name = "_bsm_g11_recovery_driver"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, driver_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the content-addressed BSM recovery driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _verify_phase_authorization(
    driver: ModuleType, record: dict[str, Any], *, contract_hash: str
) -> dict[str, Any]:
    authorization_phase = str(record["phase"])
    if (
        record.get("stage") == "fixed_family_supplement"
        and record.get("operation") == "fixed_family_supplement"
    ):
        authorization_phase = "fixed_family"
    authorization = driver._verify_execution_authorization(
        Path(str(record["execution_authorization_path"])),
        contract_hash=contract_hash,
        phase=authorization_phase,
        expected_identity=record,
    )
    if authorization.get("resource_freeze_sha256") != record.get(
        "resource_freeze_sha256"
    ):
        raise ValueError(
            "phase authorization does not bind the manifest resource freeze"
        )
    inventory_path = Path(str(record.get("campaign_inventory_path", "")))
    if not inventory_path.is_file() or hashlib.sha256(
        inventory_path.read_bytes()
    ).hexdigest() != authorization.get("campaign_inventory_hash"):
        raise ValueError(
            "phase authorization does not bind the live campaign inventory"
        )
    return authorization


def _interaction_tables(data: Any) -> tuple[Any, Any, Any, Any, Any]:
    import pandas as pd

    names = tuple(f"x{index:03d}" for index in range(158)) + (
        "binary_0",
        "binary_1",
    )
    sample_ids = tuple(f"train-{index:06d}" for index in range(data.X_train.shape[0]))
    inputs = pd.DataFrame(data.X_train, columns=names)
    inputs.insert(0, "sample_id", sample_ids)
    catalog = pd.DataFrame(
        {"feature_name": names, "feature_type": ["first_order"] * len(names)}
    )
    assignments = pd.DataFrame({"sample_id": sample_ids, "split": "train"})
    pca_scores = pd.DataFrame(
        data.Y_train,
        columns=[f"PC{index + 1}" for index in range(data.Y_train.shape[1])],
    )
    pca_scores.insert(0, "sample_id", sample_ids)
    return inputs, catalog, assignments, pca_scores, catalog.copy()


def _interaction_spec(
    campaign_contract: Any,
    *,
    draws: int,
    seed: int,
    n_jobs: int | None = None,
) -> Any:
    from rfm_pipeline.manuscript_stages import InteractionDiscoverySpec

    resolved_n_jobs = (
        max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
        if n_jobs is None
        else int(n_jobs)
    )
    if resolved_n_jobs <= 0:
        raise ValueError("interaction worker count must be positive")
    return InteractionDiscoverySpec(
        method=campaign_contract.interaction_detector_method,
        aggregation_rule="max_over_components_by_detector",
        null_threshold_quantile=0.995,
        retained_pairs_reference=0,
        permutation_count_B=draws,
        random_seed=seed,
        n_tree_estimators=campaign_contract.n_tree_estimators,
        n_jobs=resolved_n_jobs,
        selection_method=campaign_contract.method_name,
        selection_alpha=campaign_contract.alpha,
        family_partition_method=campaign_contract.family_partition_method,
        tree_family_alpha=campaign_contract.tree_family_alpha,
        binary_binary_family_alpha=campaign_contract.binary_binary_family_alpha,
        binary_binary_method=campaign_contract.binary_binary_method,
        binary_binary_minimum_cell_count=(
            campaign_contract.binary_binary_minimum_cell_count
        ),
        minimum_selection_draws=campaign_contract.B_interaction,
    )


def _load_fixed_pair_order(
    record: dict[str, Any], campaign_contract: Any
) -> tuple[str, ...]:
    path = _require_hashed_file(
        record,
        path_field="family_order_path",
        hash_field="family_order_file_sha256",
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    pairs = tuple(str(value) for value in raw)
    from rfm_pipeline.campaign_contract import fixed_family_pair_order

    if pairs != fixed_family_pair_order(campaign_contract):
        raise ValueError("fixed-family order differs from the campaign contract")
    return pairs


def _truth_support_name(term: Any, design: Any) -> str:
    """Map a typed DGP term to the exact production candidate identifier."""
    kind = str(term.term_id.kind)
    inputs = tuple(str(value) for value in term.term_id.inputs)
    if kind == "main":
        return inputs[0]
    if kind.endswith("_interaction"):
        order = {name: index for index, name in enumerate(design.input_names)}
        left, right = sorted(inputs, key=order.__getitem__)
        return f"{left}:{right}"
    if kind == "transformation":
        suffixes = {
            "quadratic": "sq",
            "logarithm": "log1p",
            "inverse": "inv",
            "square_root": "sqrt",
        }
        transform = str(term.term_id.transform)
        if transform not in suffixes:
            return term.term_id.canonical_id
        return f"{inputs[0]}_{suffixes[transform]}"
    raise ValueError(f"unsupported truth term kind {kind!r}")


def _canonical_interaction_id(value: str) -> str:
    """Return an order-invariant identifier for one undirected interaction."""
    if not isinstance(value, str):
        raise ValueError("interaction identifier must be a string")
    parts = value.split(":")
    if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
        raise ValueError(f"invalid interaction identifier {value!r}")
    return ":".join(sorted(parts))


def _canonical_support(values: Any, *, field: str) -> set[str]:
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise ValueError(f"{field} must be a sequence of support identifiers")
    canonical: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} contains an invalid support identifier")
        normalized = _canonical_interaction_id(value) if ":" in value else value
        if normalized in canonical:
            raise ValueError(f"{field} contains duplicate semantic support")
        canonical.add(normalized)
    return canonical


def _interaction_selection_outcome(row: dict[str, Any]) -> dict[str, Any]:
    truth = _canonical_support(
        row.get("truth_interaction_ids"), field="truth_interaction_ids"
    )
    retained = _canonical_support(
        row.get("retained_interaction_ids"), field="retained_interaction_ids"
    )
    if any(":" not in value for value in truth | retained):
        raise ValueError("interaction identifier fields contain a non-interaction")
    return {
        "false_pair_count": len(retained - truth),
        "planted_interaction_discovered": truth <= retained,
    }


def _support_metrics(truth: set[str], selected: set[str]) -> dict[str, Any]:
    true_positive = len(truth & selected)
    false_positive = len(selected - truth)
    return {
        "precision": true_positive / len(selected) if selected else 1.0,
        "recall": true_positive / len(truth) if truth else 1.0,
        "false_discovery_proportion": false_positive / len(selected)
        if selected
        else 0.0,
        "exact_support_recovery": selected == truth,
        "true_size": len(truth),
        "selected_size": len(selected),
    }


def _prediction_metrics(
    data: Any, predictions: dict[str, np.ndarray]
) -> dict[str, float]:
    denominators = np.ptp(np.asarray(data.Y_train, dtype=float), axis=0)
    if np.any(~np.isfinite(denominators)) or np.any(denominators <= 1.0e-12):
        raise ValueError(
            "recovery metric population has a non-finite or zero training range"
        )
    truth = np.asarray(data.Y_eval, dtype=float)
    metrics: dict[str, float] = {}
    for name, prediction in predictions.items():
        values = np.asarray(prediction, dtype=float)
        if values.shape != truth.shape or not np.isfinite(values).all():
            raise ValueError(
                f"comparator {name!r} emitted an invalid prediction matrix"
            )
        rmse = np.sqrt(np.mean((values - truth) ** 2, axis=0))
        metrics[name] = float(np.mean(rmse / denominators))
    return metrics


def _requires_recovery_comparators(scenario_kind: str) -> bool:
    if scenario_kind not in {"null", "strong", "stress"}:
        raise ValueError(f"unknown recovery scenario kind {scenario_kind!r}")
    return scenario_kind in {"strong", "stress"}


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "relative_path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _load_applied_case_config(
    record: dict[str, Any], campaign_contract: Any
) -> dict[str, Any]:
    path = _require_hashed_file(
        record,
        path_field="applied_config_path",
        hash_field="applied_config_sha256",
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("case_study"), dict):
        raise ValueError("applied case-study config is malformed")
    case_study = raw["case_study"]
    screening = case_study.get("empirical_null_screen", {})
    interaction = case_study.get("interaction_discovery", {})
    stability = case_study.get("stability", {})
    if (
        screening.get("permutation_count_source") != "g11_campaign_contract"
        or interaction.get("permutation_count_source") != "g11_campaign_contract"
    ):
        raise ValueError(
            "applied permutation schedules are not delegated to the G11 contract"
        )
    exact = {
        "B_screen": int(screening.get("permutation_count_B", -1)),
        "q_screen": float(screening.get("bh_q_screen", -1.0)),
        "interaction_detector_method": str(interaction.get("method", "")),
        "aggregation_rule": str(interaction.get("aggregation_rule", "")),
        "family_partition_method": str(
            interaction.get("family_partition_method", "")
        ),
        "tree_family_alpha": float(interaction.get("tree_family_alpha", -1.0)),
        "binary_binary_family_alpha": float(
            interaction.get("binary_binary_family_alpha", -1.0)
        ),
        "binary_binary_method": str(interaction.get("binary_binary_method", "")),
        "binary_binary_minimum_cell_count": int(
            interaction.get("binary_binary_minimum_cell_count", -1)
        ),
        "method_name": str(interaction.get("selection_method", "")),
        "alpha": float(interaction.get("selection_alpha", -1.0)),
        "B_interaction": int(interaction.get("permutation_count_B", -1)),
        "n_stability_subsamples": int(
            str(stability.get("resampling_scheme", "")).split("_", maxsplit=1)[0]
        ),
        "stability_jaccard_threshold": float(stability.get("jaccard_threshold", -1.0)),
        "stability_spearman_threshold": float(
            stability.get("spearman_threshold", -1.0)
        ),
    }
    expected = {
        "B_screen": campaign_contract.B_screen,
        "q_screen": campaign_contract.q_screen,
        "interaction_detector_method": campaign_contract.interaction_detector_method,
        "aggregation_rule": "max_over_components_by_detector",
        "family_partition_method": campaign_contract.family_partition_method,
        "tree_family_alpha": campaign_contract.tree_family_alpha,
        "binary_binary_family_alpha": campaign_contract.binary_binary_family_alpha,
        "binary_binary_method": campaign_contract.binary_binary_method,
        "binary_binary_minimum_cell_count": (
            campaign_contract.binary_binary_minimum_cell_count
        ),
        "method_name": campaign_contract.method_name,
        "alpha": campaign_contract.alpha,
        "B_interaction": campaign_contract.B_interaction,
        "n_stability_subsamples": campaign_contract.n_stability_subsamples,
        "stability_jaccard_threshold": campaign_contract.stability_jaccard_threshold,
        "stability_spearman_threshold": campaign_contract.stability_spearman_threshold,
    }
    if exact != expected:
        raise ValueError(
            "applied config scientific controls differ from the campaign contract"
        )
    return case_study


def _verify_applied_data_manifest(
    record: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    root = Path(str(record.get("applied_data_root", "")))
    manifest_path = root / "dataset_manifest.json"
    expected_hash = str(record.get("applied_data_manifest_sha256", ""))
    if expected_hash == "UNKNOWN" or len(expected_hash) != 64:
        raise ValueError("applied data manifest hash is not frozen")
    if (
        not manifest_path.is_file()
        or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_hash
    ):
        raise ValueError(
            "applied data manifest bytes differ from the campaign manifest"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = {
        key: value for key, value in payload.items() if key != "dataset_manifest_sha256"
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode()).hexdigest() != payload.get(
        "dataset_manifest_sha256"
    ):
        raise ValueError("applied data manifest self-hash differs")
    if payload.get("status") != "PREPARED_HOLDOUT_SEALED":
        raise ValueError("applied data layout is not in the sealed pre-analysis state")
    preparer_path = Path(__file__).with_name("prepare_g11_applied_data.py")
    if (
        not preparer_path.is_file()
        or payload.get("preparer_sha256")
        != hashlib.sha256(preparer_path.read_bytes()).hexdigest()
    ):
        raise ValueError(
            "applied data preparer identity differs from the prepared manifest"
        )
    dimensions = payload.get("dimensions", {})
    if dimensions != {
        "rows": 30000,
        "train_rows": 28500,
        "holdout_rows": 1500,
        "inputs": 160,
        "continuous_inputs": 158,
        "binary_inputs": 2,
        "outputs": 23495,
    }:
        raise ValueError("applied data dimensions differ from the frozen case study")
    if payload.get("binary_input_names") != [
        "FM.Use Agnostic FS Conversion",
        "OI.Use AEO Reference Oil",
    ]:
        raise ValueError(
            "applied data manifest does not identify the two binary inputs"
        )
    return root, payload


def _load_applied_train_tables(
    record: dict[str, Any],
    *,
    names: tuple[str, ...] = ("inputs", "outputs", "assignments", "catalog"),
) -> tuple[Any, ...]:
    import pandas as pd

    root, manifest = _verify_applied_data_manifest(record)
    generated = manifest["generated_sha256"]
    paths = {
        "inputs": root / "adaptive_train" / "X.parquet",
        "outputs": root / "adaptive_train" / "Y.parquet",
        "assignments": root / "adaptive_train" / "assignments.parquet",
        "catalog": root / "metadata" / "manuscript_feature_catalog.parquet",
    }
    if not names or len(names) != len(set(names)) or not set(names) <= set(paths):
        raise ValueError("requested applied training tables must be unique canonical names")
    for name in names:
        path = paths[name]
        relative = str(path.relative_to(root))
        if hashlib.sha256(path.read_bytes()).hexdigest() != generated.get(relative):
            raise ValueError(f"applied training artifact hash differs: {relative}")
    return tuple(pd.read_parquet(paths[name]) for name in names)


def _stages_root(record: dict[str, Any]) -> Path:
    return Path(str(record["output_dir"])).parents[2]


def _reducer_root(record: dict[str, Any], stage: str) -> Path:
    return _stages_root(record) / stage / "reducer"


def _read_csv(path: Path) -> Any:
    import pandas as pd

    if not path.is_file():
        raise ValueError(f"missing upstream applied artifact: {path}")
    return pd.read_csv(path)


def _load_conditioning(record: dict[str, Any]) -> Any:
    from rfm_pipeline.manuscript_stages import OutputConditioningResult

    root = _reducer_root(record, "applied_conditioning") / "output_conditioning"
    diagnostics = _read_csv(root / "output_filter_diagnostics.csv")
    return OutputConditioningResult(
        retained_output_names=tuple(
            diagnostics.loc[diagnostics["retained"].astype(bool), "output_name"].astype(
                str
            )
        ),
        culled_output_names=tuple(
            diagnostics.loc[
                ~diagnostics["retained"].astype(bool), "output_name"
            ].astype(str)
        ),
        output_filter_diagnostics=diagnostics,
        pca_scores=_read_csv(root / "pca_scores.csv"),
        pca_loadings=_read_csv(root / "pca_loadings.csv"),
        pca_explained_variance=_read_csv(root / "pca_explained_variance.csv"),
        summary=_read_csv(root / "output_conditioning_summary.csv"),
    )


def _load_screening(record: dict[str, Any]) -> Any:
    from rfm_pipeline.manuscript_stages import EmpiricalNullScreeningResult

    root = _reducer_root(record, "applied_screening") / "empirical_null_screen"
    return EmpiricalNullScreeningResult(
        feature_screening_statistics=_read_csv(
            root / "feature_screening_statistics.csv"
        ),
        component_coefficients=_read_csv(root / "component_coefficients.csv"),
        permutation_null_summary=_read_csv(root / "permutation_null_summary.csv"),
        retained_terms=_read_csv(root / "retained_terms.csv"),
        provenance=_read_csv(root / "empirical_null_provenance.csv"),
        summary=_read_csv(root / "empirical_null_screen_summary.csv"),
    )


def _load_interactions(record: dict[str, Any]) -> Any:
    from rfm_pipeline.manuscript_stages import InteractionDiscoveryResult

    root = _reducer_root(record, "applied_interaction") / "interaction_discovery"
    return InteractionDiscoveryResult(
        pair_scores=_read_csv(root / "interaction_pair_scores.csv"),
        component_interaction_scores=_read_csv(
            root / "component_interaction_scores.csv"
        ),
        interaction_null_summary=_read_csv(root / "interaction_null_summary.csv"),
        retained_pairs=_read_csv(root / "retained_interaction_pairs.csv"),
        provenance=_read_csv(root / "interaction_discovery_provenance.csv"),
        summary=_read_csv(root / "interaction_discovery_summary.csv"),
    )


def _load_nonlinear(record: dict[str, Any]) -> Any:
    from rfm_pipeline.manuscript_stages import NonlinearDiscoveryResult

    root = _reducer_root(record, "applied_nonlinear") / "nonlinear_discovery"
    return NonlinearDiscoveryResult(
        transformation_scores=_read_csv(root / "transformation_scores.csv"),
        component_transformation_scores=_read_csv(
            root / "component_transformation_scores.csv"
        ),
        retained_transformations=_read_csv(root / "retained_transformations.csv"),
        provenance=_read_csv(root / "nonlinear_discovery_provenance.csv"),
        summary=_read_csv(root / "nonlinear_discovery_summary.csv"),
    )


def _load_sparse(record: dict[str, Any]) -> Any:
    from rfm_pipeline.manuscript_stages import SparseSelectionStabilityResult

    root = _reducer_root(record, "applied_sparse_resample") / "sparse_selection"
    return SparseSelectionStabilityResult(
        support_candidates=_read_csv(root / "support_candidates.csv"),
        component_model_selection=_read_csv(root / "component_model_selection.csv"),
        component_coefficients=_read_csv(root / "component_coefficients.csv"),
        stability_resample_summary=_read_csv(root / "stability_resample_summary.csv"),
        stability_feature_summary=_read_csv(root / "stability_feature_summary.csv"),
        final_stable_support=_read_csv(root / "final_stable_support.csv"),
        provenance=_read_csv(root / "sparse_selection_provenance.csv"),
        summary=_read_csv(root / "sparse_selection_summary.csv"),
    )


def _write_terminal_record(shard_dir: Path, payload: dict[str, Any]) -> Path:
    path = shard_dir / "terminal_record.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _score_artifact_payload_sha256(artifact: Any) -> str:
    """Return the canonical score-block identity without a compatibility alias."""
    payload_sha256 = str(getattr(artifact, "payload_sha256", ""))
    if len(payload_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in payload_sha256
    ):
        raise ValueError(
            "score-only interaction artifact has no valid payload identity"
        )
    return payload_sha256


def _load_reduction_contract(records: list[dict[str, Any]]) -> tuple[Any, str]:
    from rfm_pipeline.campaign_contract import load_contract

    paths = {str(record.get("contract_config_path", "")) for record in records}
    hashes = {str(record.get("config_hash", "")) for record in records}
    if len(paths) != 1 or "" in paths or len(hashes) != 1 or "" in hashes:
        raise ValueError(
            "scientific reduction requires one manifest-bound contract identity"
        )
    contract, contract_hash = load_contract(Path(paths.pop()))
    if hashes != {contract_hash}:
        raise ValueError("scientific reduction contract hash differs from its records")
    return contract, contract_hash


def _read_terminal_payloads(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("scientific reduction requires at least one record")
    operation = str(records[0].get("operation", ""))
    payloads: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("operation", "")) != operation:
            raise ValueError("scientific reduction received mixed operations")
        shard_dir = Path(str(record.get("output_dir", "")))
        result_path = shard_dir / "result.json"
        terminal_path = shard_dir / "terminal_record.json"
        if not result_path.is_file() or not terminal_path.is_file():
            raise ValueError(f"missing terminal record for {record.get('shard_id')}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("operation") != operation
            or result.get("status") != "completed"
            or Path(str(result.get("terminal_record", ""))).resolve()
            != terminal_path.resolve()
        ):
            raise ValueError(
                f"invalid terminal result identity for {record.get('shard_id')}"
            )
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        if (
            terminal.get("operation") != operation
            or terminal.get("status") != "completed"
        ):
            raise ValueError(f"invalid terminal payload for {record.get('shard_id')}")
        payloads.append(terminal)
    return payloads


def _write_reduction(
    output_dir: Path,
    *,
    name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        **payload,
        "artifact_path": str(path),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scientific_artifacts": _artifact_inventory(output_dir),
    }


def _reduce_resolution(
    records: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    from rfm_pipeline.campaign_contract import select_resolution_draw_count

    contract, contract_hash = _load_reduction_contract(records)
    decision = select_resolution_draw_count(payloads, contract)
    resolution_status = str(decision.pop("status"))
    return _write_reduction(
        output_dir,
        name="resolution_decision.json",
        payload={
            **decision,
            "operation": "resolution",
            "status": "completed",
            "decision": resolution_status,
            "contract_hash": contract_hash,
            "terminal_record_count": len(payloads),
        },
    )


def _wilson_lower_bound(events: int, denominator: int, confidence: float) -> float:
    from rfm_pipeline.campaign_contract import wilson_upper_bound

    return 1.0 - wilson_upper_bound(denominator - events, denominator, confidence)


def _reduce_gate_b(
    records: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    from rfm_pipeline.campaign_contract import wilson_upper_bound

    contract, contract_hash = _load_reduction_contract(records)
    summaries: dict[str, dict[str, Any]] = {}
    passed = True
    for scenario in (
        item for item in contract.scenarios if item.kind in {"null", "strong"}
    ):
        rows = [row for row in payloads if row.get("scenario") == scenario.id]
        indices = [int(row.get("replicate_index", -1)) for row in rows]
        if sorted(indices) != list(range(scenario.n_replicates)):
            raise ValueError(f"Gate B does not exactly cover scenario {scenario.id}")
        outcomes = [_interaction_selection_outcome(row) for row in rows]
        if scenario.kind == "null":
            events = sum(
                int(outcome["false_pair_count"]) > 0 for outcome in outcomes
            )
            upper = wilson_upper_bound(
                events,
                scenario.n_replicates,
                contract.calibration_confidence,
            )
            summary = {
                "denominator": scenario.n_replicates,
                "false_selection_events": events,
                "fwer": events / scenario.n_replicates,
                "one_sided_wilson_upper": upper,
                "passes_calibration": upper <= contract.gate_value,
            }
            if scenario.id != "global_null":
                nondegenerate = sum(int(row["pair_family_count"]) >= 2 for row in rows)
                summary.update(
                    {
                        "nondegenerate_count": nondegenerate,
                        "nondegenerate_rate": nondegenerate / scenario.n_replicates,
                        "passes_nondegeneracy": nondegenerate / scenario.n_replicates
                        >= 0.90,
                    }
                )
                passed &= bool(summary["passes_nondegeneracy"])
            passed &= bool(summary["passes_calibration"])
        else:
            events = sum(
                bool(outcome["planted_interaction_discovered"])
                for outcome in outcomes
            )
            lower = _wilson_lower_bound(
                events,
                scenario.n_replicates,
                contract.calibration_confidence,
            )
            summary = {
                "denominator": scenario.n_replicates,
                "discovery_events": events,
                "power": events / scenario.n_replicates,
                "one_sided_wilson_lower": lower,
                "passes_power": lower >= contract.power_gate_lower_bound,
            }
            passed &= bool(summary["passes_power"])
        summaries[scenario.id] = summary
    if not passed:
        raise ValueError(
            "Gate B failed a prespecified calibration, power, or nondegeneracy rule"
        )
    return _write_reduction(
        output_dir,
        name="gate_b_decision.json",
        payload={
            "operation": "gate_b",
            "status": "completed",
            "decision": "PASS",
            "contract_hash": contract_hash,
            "terminal_record_count": len(payloads),
            "scenarios": summaries,
        },
    )


def _reduce_fixed_family(
    records: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    from rfm_pipeline.campaign_contract import wilson_upper_bound

    contract, contract_hash = _load_reduction_contract(records)
    indices = [int(row.get("global_null_replicate_index", -1)) for row in payloads]
    if sorted(indices) != list(range(contract.fixed_family_replicates)):
        raise ValueError("fixed-family reduction lacks exact replicate coverage")
    summaries: dict[str, dict[str, Any]] = {}
    for family_size in contract.fixed_family_sizes:
        key = str(family_size)
        events = sum(bool(row["selected_pairs"][key]) for row in payloads)
        upper = wilson_upper_bound(
            events,
            contract.fixed_family_replicates,
            contract.calibration_confidence,
        )
        summaries[key] = {
            "denominator": contract.fixed_family_replicates,
            "false_selection_events": events,
            "fwer": events / contract.fixed_family_replicates,
            "one_sided_wilson_upper": upper,
            "passes_calibration": upper <= contract.gate_value,
        }
    if not all(row["passes_calibration"] for row in summaries.values()):
        raise ValueError(
            "fixed-family supplement failed its prespecified calibration rule"
        )
    return _write_reduction(
        output_dir,
        name="fixed_family_decision.json",
        payload={
            "operation": "fixed_family_supplement",
            "status": "completed",
            "decision": "PASS",
            "contract_hash": contract_hash,
            "terminal_record_count": len(payloads),
            "families": summaries,
        },
    )


def _reduce_recovery(
    records: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    contract, contract_hash = _load_reduction_contract(records)
    gate_b_paths = {str(record.get("gate_b_manifest_path", "")) for record in records}
    if len(gate_b_paths) != 1 or "" in gate_b_paths:
        raise ValueError("Gate C reduction requires one immutable Gate-B manifest")
    gate_b_manifest = Path(gate_b_paths.pop())
    if not gate_b_manifest.is_file():
        raise ValueError("Gate C reduction cannot read its Gate-B manifest")
    gate_b_records = [
        json.loads(line)
        for line in gate_b_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(record.get("config_hash") != contract_hash for record in gate_b_records):
        raise ValueError("Gate C Gate-B manifest has a different contract identity")
    strong_payloads = [
        payload
        for payload in _read_terminal_payloads(gate_b_records)
        if payload.get("scenario")
        in {scenario.id for scenario in contract.scenarios if scenario.kind == "strong"}
    ]
    combined = strong_payloads + payloads
    expected = {
        (scenario.id, index)
        for scenario in contract.scenarios
        if scenario.kind in {"strong", "stress"}
        for index in range(scenario.n_replicates)
    }
    actual = {
        (str(row.get("scenario")), int(row.get("replicate_index", -1)))
        for row in combined
    }
    if actual != expected:
        raise ValueError(
            "Gate C stress records do not exactly cover the frozen campaign"
        )
    return _write_reduction(
        output_dir,
        name="gate_c_stress_ledger_summary.json",
        payload={
            "operation": "recovery",
            "status": "completed",
            "acceptance_status": "READY_FOR_INDEPENDENT_REVIEW",
            "contract_hash": contract_hash,
            "terminal_record_count": len(combined),
            "reused_gate_b_strong_record_count": len(strong_payloads),
            "new_stress_record_count": len(payloads),
        },
    )


def _applied_reduction_result(
    operation: str, output_dir: Path, **values: Any
) -> dict[str, Any]:
    payload = {"operation": operation, "status": "completed", **values}
    path = output_dir / "scientific_stage_summary.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**payload, "scientific_artifacts": _artifact_inventory(output_dir)}


def _reduce_applied_conditioning(
    records: list[dict[str, Any]], payloads: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    if len(records) != 1 or len(payloads) != 1:
        raise ValueError("applied conditioning requires exactly one completed worker")
    source = Path(records[0]["output_dir"]) / "output_conditioning"
    shutil.copytree(source, output_dir / "output_conditioning")
    return _applied_reduction_result(
        "applied_conditioning",
        output_dir,
        retained_output_count=int(payloads[0]["retained_output_count"]),
        culled_output_count=int(payloads[0]["culled_output_count"]),
    )


def _reduce_applied_screening(
    records: list[dict[str, Any]], payloads: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    from rfm_pipeline.manuscript_stages import (
        ScreeningNullBlock,
        reduce_screening_draw_blocks,
        write_empirical_null_screening_artifacts,
    )

    contract, _ = _load_reduction_contract(records)
    _, screening, *_ = _applied_specifications(records[0], contract)
    blocks = [
        ScreeningNullBlock.read(Path(record["output_dir"]) / "screening_block")
        for record in records
    ]
    result = reduce_screening_draw_blocks(blocks, spec=screening)
    write_empirical_null_screening_artifacts(result, output_dir)
    return _applied_reduction_result(
        "applied_screening",
        output_dir,
        retained_term_count=len(result.retained_terms),
        draw_count=contract.B_screen,
    )


def _reduce_applied_interaction(
    records: list[dict[str, Any]], payloads: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    from itertools import combinations

    from rfm_pipeline.interaction_contract import (
        ScoreOnlyInteractionArtifact,
        canonical_execution_contract_from_specs,
    )
    from rfm_pipeline.manuscript_stages import (
        reduce_score_only_interaction_artifacts,
        write_interaction_discovery_artifacts,
    )

    contract, _ = _load_reduction_contract(records)
    screening = _load_screening(records[0])
    _, _, interaction, *_ = _applied_specifications(records[0], contract)
    retained_names = screening.retained_terms["feature_name"].astype(str).tolist()
    expected_pairs = tuple(
        f"{left}:{right}" for left, right in combinations(retained_names, 2)
    )
    artifacts = [
        ScoreOnlyInteractionArtifact.read_from(
            Path(record["output_dir"]) / "interaction_block"
        )
        for record in records
    ]
    if artifacts and not artifacts[0].pair_names:
        reference_checksum = artifacts[0].control_snapshot.checksum
        if any(
            artifact.status != "empty_candidate_family"
            or artifact.pair_names
            or artifact.control_snapshot.checksum != reference_checksum
            for artifact in artifacts
        ):
            raise ValueError(
                "empty applied interaction workers have inconsistent identities"
            )
        artifacts = [artifacts[0]]
    result = reduce_score_only_interaction_artifacts(
        artifacts,
        spec=interaction,
        contract=canonical_execution_contract_from_specs(interaction),
        expected_pair_names=expected_pairs,
    )
    write_interaction_discovery_artifacts(result, output_dir)
    return _applied_reduction_result(
        "applied_interaction",
        output_dir,
        candidate_pair_count=len(result.pair_scores),
        retained_pair_count=len(result.retained_pairs),
        draw_count=contract.B_interaction,
    )


def _reduce_applied_nonlinear(
    records: list[dict[str, Any]], payloads: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    from rfm_pipeline.manuscript_stages import (
        _generate_supported_nonlinear_candidates,
        discover_manuscript_nonlinear_transformations,
        write_nonlinear_discovery_artifacts,
    )

    contract, _ = _load_reduction_contract(records)
    checkpoint_root = output_dir / "checkpoints"
    for record in records:
        source = Path(record["output_dir"]) / "checkpoints"
        if not source.exists():
            continue
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            destination = checkpoint_root / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (
                destination.exists()
                and hashlib.sha256(destination.read_bytes()).hexdigest()
                != hashlib.sha256(path.read_bytes()).hexdigest()
            ):
                if path.name != "checkpoint_metadata.json":
                    raise ValueError(
                        "nonlinear checkpoint collision has different bytes"
                    )
                existing = json.loads(destination.read_text(encoding="utf-8"))
                incoming = json.loads(path.read_text(encoding="utf-8"))
                existing.pop("created_at_utc", None)
                incoming.pop("created_at_utc", None)
                if existing != incoming:
                    raise ValueError("nonlinear checkpoint metadata identities differ")
            if not destination.exists():
                shutil.copy2(path, destination)
    inputs, assignments, catalog = _load_applied_train_tables(
        records[0], names=("inputs", "assignments", "catalog")
    )
    conditioning = _load_conditioning(records[0])
    screening = _load_screening(records[0])
    _, _, _, nonlinear, *_ = _applied_specifications(records[0], contract)
    retained_names = screening.retained_terms["feature_name"].astype(str).tolist()
    base_names = tuple(
        dict.fromkeys(
            base
            for _, base, _ in _generate_supported_nonlinear_candidates(
                retained_names,
                inputs,
                transform_library=nonlinear.transform_library,
            )
        )
    )
    run_dirs = [path for path in checkpoint_root.iterdir() if path.is_dir()]
    if len(run_dirs) != 1:
        raise ValueError(
            "nonlinear reduction requires one complete checkpoint identity"
        )
    expected_names = {f"feature_{index:06d}.json" for index in range(len(base_names))}
    actual_names = {
        path.name for path in run_dirs[0].glob("feature_*.json") if path.is_file()
    }
    if actual_names != expected_names:
        raise ValueError(
            "nonlinear reduction lacks exact base-feature checkpoint coverage"
        )
    for index, base_name in enumerate(base_names):
        checkpoint = json.loads(
            (run_dirs[0] / f"feature_{index:06d}.json").read_text(encoding="utf-8")
        )
        if (
            checkpoint.get("base_feature") != base_name
            or not isinstance(checkpoint.get("feature_result"), dict)
            or not isinstance(checkpoint.get("cache_entries"), list)
        ):
            raise ValueError("nonlinear reduction found a malformed feature checkpoint")
    result = discover_manuscript_nonlinear_transformations(
        inputs,
        catalog,
        assignments,
        conditioning.pca_scores,
        screening.retained_terms,
        nonlinear,
        checkpoint_dir=checkpoint_root,
    )
    write_nonlinear_discovery_artifacts(result, output_dir)
    return _applied_reduction_result(
        "applied_nonlinear",
        output_dir,
        retained_transformation_count=len(result.retained_transformations),
    )


def _reduce_applied_sparse_full(
    records: list[dict[str, Any]], payloads: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    from rfm_pipeline.manuscript_stages import SparseFullFitArtifact

    if len(records) != 1:
        raise ValueError("applied sparse full fit requires exactly one worker")
    source = Path(records[0]["output_dir"]) / "sparse_full_fit"
    artifact = SparseFullFitArtifact.read(source)
    shutil.copytree(source, output_dir / "sparse_full_fit")
    return _applied_reduction_result(
        "applied_sparse_full",
        output_dir,
        full_fit_hash=artifact.artifact_hash,
        candidate_count=len(artifact.candidate_names),
    )


def _reduce_applied_sparse_resample(
    records: list[dict[str, Any]], payloads: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    from rfm_pipeline.manuscript_stages import (
        SparseFullFitArtifact,
        SparseStabilityBlock,
        reduce_sparse_stability_blocks,
        write_sparse_selection_stability_artifacts,
    )

    contract, _ = _load_reduction_contract(records)
    *_, sparse, _ = _applied_specifications(records[0], contract)
    full_fit = SparseFullFitArtifact.read(
        _reducer_root(records[0], "applied_sparse_full") / "sparse_full_fit"
    )
    blocks = [
        SparseStabilityBlock.read(path)
        for record in records
        for path in sorted(
            (Path(record["output_dir"]) / "sparse_stability_blocks").glob("resample-*")
        )
    ]
    if len(blocks) != contract.n_stability_subsamples:
        raise ValueError("sparse stability blocks do not exactly cover all resamples")
    (catalog,) = _load_applied_train_tables(records[0], names=("catalog",))
    screening = _load_screening(records[0])
    interactions = _load_interactions(records[0])
    nonlinear = _load_nonlinear(records[0])
    result = reduce_sparse_stability_blocks(
        full_fit,
        blocks,
        feature_catalog=catalog,
        retained_terms=screening.retained_terms,
        retained_interaction_pairs=interactions.retained_pairs,
        retained_transformations=nonlinear.retained_transformations,
        spec=sparse,
    )
    write_sparse_selection_stability_artifacts(result, output_dir)
    return _applied_reduction_result(
        "applied_sparse_resample",
        output_dir,
        final_stable_support_count=len(result.final_stable_support),
        resample_count=len(blocks),
    )


def _reduce_applied_terminal_fit(
    records: list[dict[str, Any]], payloads: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    if len(records) != 1:
        raise ValueError("applied terminal fit requires exactly one worker")
    shutil.copytree(
        Path(records[0]["output_dir"]) / "terminal", output_dir / "terminal"
    )
    return _applied_reduction_result(
        "applied_terminal_fit",
        output_dir,
        freeze_hash=str(payloads[0]["freeze_hash"]),
        final_count=int(payloads[0]["final_count"]),
    )


def _reduce_applied_ablation_fit(
    records: list[dict[str, Any]], payloads: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    by_model = {
        str(payload["ablation_model"]): (record, payload)
        for record, payload in zip(records, payloads, strict=True)
    }
    if set(by_model) != set(_ABLATION_MODELS):
        raise ValueError("applied ablation workers do not cover the five frozen models")
    freeze_hashes = {}
    for model in _ABLATION_MODELS:
        record, payload = by_model[model]
        shutil.copytree(
            Path(record["output_dir"]) / "model", output_dir / "models" / model
        )
        freeze_hashes[model] = str(payload["freeze_hash"])
    return _applied_reduction_result(
        "applied_ablation_fit",
        output_dir,
        model_freeze_hashes=freeze_hashes,
    )


def _reduce_applied_holdout_authorize(
    records: list[dict[str, Any]], payloads: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    if len(records) != 1:
        raise ValueError("holdout authorization requires exactly one worker")
    source = Path(records[0]["output_dir"]) / "holdout_access_authorization.json"
    shutil.copy2(source, output_dir / source.name)
    return _applied_reduction_result(
        "applied_holdout_authorize",
        output_dir,
        model_freeze_hashes=payloads[0]["model_freeze_hashes"],
    )


def _reduce_applied_holdout_predict(
    records: list[dict[str, Any]], payloads: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    import pandas as pd

    from rfm_pipeline.manuscript_stages import (
        FrozenPredictionMatrices,
        read_frozen_prediction_matrices,
        write_frozen_prediction_matrices,
    )

    ordered = sorted(
        zip(records, payloads, strict=True),
        key=lambda value: int(value[1]["output_start"]),
    )
    expected = 0
    for _, payload in ordered:
        if int(payload["output_start"]) != expected:
            raise ValueError("holdout prediction blocks have a gap or overlap")
        expected = int(payload["output_end"])
    if expected != 23495:
        raise ValueError("holdout prediction blocks do not cover 23,495 outputs")
    freeze_hashes = {}
    for model in _ABLATION_MODELS:
        blocks = [
            read_frozen_prediction_matrices(
                Path(record["output_dir"]) / "models" / model
            )
            for record, _ in ordered
        ]
        reference = blocks[0]
        if any(
            block.freeze_hash != reference.freeze_hash
            or block.truth_ids != reference.truth_ids
            or block.strata != reference.strata
            for block in blocks[1:]
        ):
            raise ValueError(f"holdout prediction block identity differs for {model}")
        output_names = tuple(name for block in blocks for name in block.output_names)
        if len(output_names) != 23495 or len(set(output_names)) != 23495:
            raise ValueError(
                f"holdout prediction output IDs are incomplete for {model}"
            )
        combined = FrozenPredictionMatrices(
            freeze_hash=reference.freeze_hash,
            y_holdout=np.concatenate([block.y_holdout for block in blocks], axis=1),
            y_pred=np.concatenate([block.y_pred for block in blocks], axis=1),
            y_train_min=np.concatenate([block.y_train_min for block in blocks]),
            y_train_max=np.concatenate([block.y_train_max for block in blocks]),
            y_train_mean=np.concatenate([block.y_train_mean for block in blocks]),
            y_train_variance=np.concatenate(
                [block.y_train_variance for block in blocks]
            ),
            output_names=output_names,
            truth_ids=reference.truth_ids,
            prediction_ids=reference.prediction_ids,
            strata=reference.strata,
            eligibility_ledger=pd.concat(
                [block.eligibility_ledger for block in blocks],
                ignore_index=True,
            ),
        )
        write_frozen_prediction_matrices(
            frozen=combined,
            output_dir=output_dir / "models" / model,
        )
        freeze_hashes[model] = combined.freeze_hash
    return _applied_reduction_result(
        "applied_holdout_predict",
        output_dir,
        output_count=23495,
        model_freeze_hashes=freeze_hashes,
    )


def _reduce_applied_eligibility(
    records: list[dict[str, Any]], payloads: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    if len(records) != 1:
        raise ValueError("applied eligibility requires exactly one worker")
    source = Path(records[0]["output_dir"]) / "output_eligibility_ledger.csv"
    shutil.copy2(source, output_dir / source.name)
    return _applied_reduction_result(
        "applied_eligibility",
        output_dir,
        accounting=payloads[0]["accounting"],
        freeze_hash=payloads[0]["freeze_hash"],
    )


def _read_bootstrap_block(path: Path) -> Any:
    from rfm_pipeline.manuscript_stages import BootstrapMetricShard

    metadata = json.loads((path / "bootstrap_block.json").read_text(encoding="utf-8"))
    arrays_path = path / "bootstrap_block.npz"
    if hashlib.sha256(arrays_path.read_bytes()).hexdigest() != metadata["npz_sha256"]:
        raise ValueError("bootstrap block array checksum differs")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        return BootstrapMetricShard(
            freeze_hash=str(metadata["freeze_hash"]),
            schedule_hash=str(metadata["schedule_hash"]),
            draw_start=int(metadata["draw_start"]),
            draw_end=int(metadata["draw_end"]),
            per_draw_macro_nrmse=np.asarray(
                arrays["per_draw_macro_nrmse"], dtype=float
            ),
            eligible_count=int(metadata["eligible_count"]),
            draw_ids=np.asarray(arrays["draw_ids"], dtype=np.int64),
            stratum_counts=tuple(int(value) for value in metadata["stratum_counts"]),
        )


def _reduce_applied_bootstrap(
    records: list[dict[str, Any]], payloads: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    from itertools import combinations

    from rfm_pipeline.manuscript_stages import (
        read_frozen_prediction_matrices,
        reduce_bootstrap_metric_blocks,
    )

    contract, contract_hash = _load_reduction_contract(records)
    summary: dict[str, dict[str, Any]] = {}
    draw_values: dict[str, np.ndarray] = {}
    point_estimates: dict[str, float] = {}
    output_dir.joinpath("models").mkdir(parents=True)
    for model in _ABLATION_MODELS:
        blocks = [
            _read_bootstrap_block(Path(record["output_dir"]) / "models" / model)
            for record in records
        ]
        combined = reduce_bootstrap_metric_blocks(
            blocks,
            expected_total_draws=contract.bootstrap_draws,
        )
        values = combined.per_draw_macro_nrmse
        if not np.isfinite(values).all() or combined.eligible_count != 9954:
            raise ValueError(f"bootstrap reduction is invalid for {model}")
        frozen = read_frozen_prediction_matrices(
            _reducer_root(records[0], "applied_holdout_predict") / "models" / model
        )
        eligible = frozen.eligibility_ledger["eligible"].to_numpy(dtype=bool)
        ranges = frozen.y_train_max - frozen.y_train_min
        observed_rmse = np.sqrt(
            np.mean((frozen.y_holdout - frozen.y_pred) ** 2, axis=0)
        )
        observed = float(np.mean((observed_rmse / ranges)[eligible]))
        if not np.isfinite(observed):
            raise ValueError(f"observed macro nRMSE is invalid for {model}")
        model_summary = {
            "model": model,
            "metric": "macro_nrmse",
            "point_estimate": observed,
            "ci_lower": float(np.quantile(values, 0.025)),
            "ci_upper": float(np.quantile(values, 0.975)),
            "draw_count": len(values),
            "eligible_count": combined.eligible_count,
            "freeze_hash": combined.freeze_hash,
            "schedule_hash": combined.schedule_hash,
        }
        path = output_dir / "models" / f"{model}.json"
        path.write_text(
            json.dumps(model_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary[model] = model_summary
        draw_values[model] = values
        point_estimates[model] = observed
    contrasts: dict[str, dict[str, Any]] = {}
    for left, right in combinations(_ABLATION_MODELS, 2):
        difference = draw_values[left] - draw_values[right]
        name = f"{left}_minus_{right}"
        contrasts[name] = {
            "left_model": left,
            "right_model": right,
            "metric": "paired_macro_nrmse_difference",
            "point_estimate": point_estimates[left] - point_estimates[right],
            "ci_lower": float(np.quantile(difference, 0.025)),
            "ci_upper": float(np.quantile(difference, 0.975)),
            "draw_count": len(difference),
        }
    contrast_path = output_dir / "paired_contrasts.json"
    contrast_path.write_text(
        json.dumps(contrasts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return _applied_reduction_result(
        "applied_bootstrap",
        output_dir,
        contract_hash=contract_hash,
        draw_count=contract.bootstrap_draws,
        eligible_count=9954,
        models=summary,
        paired_contrasts=contrasts,
    )


def _reduce_applied_stage(
    operation: str,
    records: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    reducers = {
        "applied_conditioning": _reduce_applied_conditioning,
        "applied_screening": _reduce_applied_screening,
        "applied_interaction": _reduce_applied_interaction,
        "applied_nonlinear": _reduce_applied_nonlinear,
        "applied_sparse_full": _reduce_applied_sparse_full,
        "applied_sparse_resample": _reduce_applied_sparse_resample,
        "applied_terminal_fit": _reduce_applied_terminal_fit,
        "applied_ablation_fit": _reduce_applied_ablation_fit,
        "applied_holdout_authorize": _reduce_applied_holdout_authorize,
        "applied_holdout_predict": _reduce_applied_holdout_predict,
        "applied_eligibility": _reduce_applied_eligibility,
        "applied_bootstrap": _reduce_applied_bootstrap,
    }
    reducer = reducers.get(operation)
    if reducer is None:
        raise ValueError(f"unknown applied reducer {operation!r}")
    return reducer(records, payloads, output_dir)


def reduce_scientific_stage(
    records: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    """Reduce one exact production stage and enforce its scientific gate."""
    payloads = _read_terminal_payloads(records)
    operation = str(records[0]["operation"])
    if operation.startswith("applied_"):
        return _reduce_applied_stage(operation, records, payloads, output_dir)
    reducers = {
        "resolution": _reduce_resolution,
        "gate_b": _reduce_gate_b,
        "fixed_family_supplement": _reduce_fixed_family,
        "recovery": _reduce_recovery,
    }
    reducer = reducers.get(operation)
    if reducer is None:
        raise ValueError(f"BSM campaign adapter does not reduce {operation!r}")
    return reducer(records, payloads, output_dir)


def _resolution_terminal_from_artifact(
    *, record: dict[str, Any], artifact: Any, campaign_contract: Any
) -> dict[str, Any]:
    """Materialize the resolution decision surface from one completed score block."""
    from rfm_pipeline.manuscript_stages import max_t_adjusted_pvalues

    pair_order = _load_fixed_pair_order(record, campaign_contract)[
        : int(record["family_size"])
    ]
    max_draws = int(record["nested_schedule_draws"])
    if (
        tuple(artifact.pair_names) != pair_order
        or int(artifact.draw_range_start) != 0
        or int(artifact.draw_range_end) != max_draws
        or artifact.observed_scores.shape != (len(pair_order),)
        or artifact.null_scores.shape != (max_draws, len(pair_order))
    ):
        raise ValueError(
            "cached resolution score block differs from its manifest work unit"
        )
    multipliers = (
        1,
        2,
        max_draws // int(record["base_draws"]),
    )
    draw_schedule = tuple(int(record["base_draws"]) * value for value in multipliers)
    if draw_schedule[-1] != max_draws or len(set(draw_schedule)) != len(draw_schedule):
        raise ValueError(
            "resolution manifest does not define three nested draw schedules"
        )
    decisions: dict[str, list[str]] = {}
    adjusted_p_values: dict[str, list[float]] = {}
    for draws in draw_schedule:
        adjusted = max_t_adjusted_pvalues(
            artifact.observed_scores,
            artifact.null_scores[:draws],
        )
        adjusted_p_values[str(draws)] = adjusted.tolist()
        decisions[str(draws)] = [
            pair
            for pair, p_value in zip(pair_order, adjusted, strict=True)
            if p_value <= campaign_contract.alpha
        ]
    return {
        "operation": "resolution",
        "fixture_kind": str(record["fixture_kind"]),
        "schedule_index": int(record["schedule_index"]),
        "family_size": len(pair_order),
        "nested_draws": list(draw_schedule),
        "selected_pairs": decisions,
        "adjusted_p_values": adjusted_p_values,
        "artifact_checksum": _score_artifact_payload_sha256(artifact),
        "status": "completed",
    }


def _execute_resolution(
    *,
    record: dict[str, Any],
    shard_dir: Path,
    campaign_contract: Any,
    driver: ModuleType,
    dgp_contract: Any,
    design: Any,
) -> dict[str, Any]:
    from rfm_pipeline.interaction_contract import (
        canonical_execution_contract_from_specs,
    )
    from rfm_pipeline.manuscript_stages import score_interaction_draw_block

    scenario_id = (
        "interaction_null_binary_main"
        if record["fixture_kind"] == "nondegenerate_null"
        else "strong_cc"
    )
    data = driver.generate_campaign_dataset(
        campaign_contract=campaign_contract,
        bsm_contract=dgp_contract,
        design=design,
        scenario_id=scenario_id,
        replicate_index=int(record["schedule_index"]),
        scale=driver.DatasetScale(n_train=160, n_eval=80, n_outputs=4),
    )
    pair_order = _load_fixed_pair_order(record, campaign_contract)[
        : int(record["family_size"])
    ]
    max_draws = int(record["nested_schedule_draws"])
    spec = _interaction_spec(
        campaign_contract,
        draws=max_draws,
        seed=int(record["seed"]),
    )
    artifact = score_interaction_draw_block(
        *_interaction_tables(data),
        spec,
        draw_start=0,
        draw_end=max_draws,
        contract=canonical_execution_contract_from_specs(spec),
        candidate_pair_names=pair_order,
    )
    artifact_dir = shard_dir / "interaction_score_blocks"
    artifact.write_to(artifact_dir)
    terminal = _resolution_terminal_from_artifact(
        record=record,
        artifact=artifact,
        campaign_contract=campaign_contract,
    )
    terminal_path = _write_terminal_record(shard_dir, terminal)
    return {
        **terminal,
        "terminal_record": str(terminal_path),
        "scientific_artifacts": _artifact_inventory(shard_dir),
    }


def _select_fixed_family_pairs(
    *,
    observed_scores: np.ndarray,
    null_scores: np.ndarray,
    pair_order: tuple[str, ...],
    pair_detectors: tuple[str, ...],
    family_sizes: tuple[int, ...],
    tree_family_alpha: float,
    binary_binary_family_alpha: float,
) -> dict[str, list[str]]:
    """Select nested fixed families inside the frozen detector partitions."""
    from rfm_pipeline.manuscript_stages import max_t_adjusted_pvalues

    observed = np.asarray(observed_scores, dtype=float)
    null = np.asarray(null_scores, dtype=float)
    if observed.ndim != 1 or null.ndim != 2 or null.shape[1] != observed.size:
        raise ValueError("fixed-family observed/null score shapes are inconsistent")
    if len(pair_order) != observed.size or len(pair_detectors) != observed.size:
        raise ValueError("fixed-family pair and detector identities must cover every score")
    if not np.isfinite(observed).all() or not np.isfinite(null).all():
        raise ValueError("fixed-family selection rejects non-finite scores")
    alpha_by_detector = {
        "tree_shap": float(tree_family_alpha),
        "studentized_binary_factorial": float(binary_binary_family_alpha),
    }
    if any(not 0.0 < alpha < 1.0 for alpha in alpha_by_detector.values()):
        raise ValueError("fixed-family detector alphas must lie in the open interval (0, 1)")

    selected_by_family: dict[str, list[str]] = {}
    for family_size in family_sizes:
        size = int(family_size)
        if size <= 0 or size > observed.size:
            raise ValueError("fixed-family size is outside the persisted candidate family")
        prefix_detectors = pair_detectors[:size]
        unknown = sorted(set(prefix_detectors).difference(alpha_by_detector))
        if unknown:
            raise ValueError(f"fixed-family selection found unsupported detectors: {unknown}")
        retained = np.zeros(size, dtype=bool)
        for detector, alpha in alpha_by_detector.items():
            indices = np.asarray(
                [index for index, value in enumerate(prefix_detectors) if value == detector],
                dtype=int,
            )
            if indices.size == 0:
                continue
            adjusted = max_t_adjusted_pvalues(
                observed[indices],
                null[:, indices],
            )
            retained[indices] = adjusted <= alpha
        selected_by_family[str(size)] = [
            pair for pair, keep in zip(pair_order[:size], retained, strict=True) if keep
        ]
    return selected_by_family


def _execute_fixed_family(
    *,
    record: dict[str, Any],
    shard_dir: Path,
    campaign_contract: Any,
    driver: ModuleType,
    dgp_contract: Any,
    design: Any,
) -> dict[str, Any]:
    from rfm_pipeline.interaction_contract import (
        canonical_execution_contract_from_specs,
    )
    from rfm_pipeline.manuscript_stages import score_interaction_draw_block

    data = driver.generate_campaign_dataset(
        campaign_contract=campaign_contract,
        bsm_contract=dgp_contract,
        design=design,
        scenario_id="global_null",
        replicate_index=int(record["global_null_replicate_index"]),
    )
    pair_order = _load_fixed_pair_order(record, campaign_contract)
    spec = _interaction_spec(
        campaign_contract,
        draws=campaign_contract.B_interaction,
        seed=int(record["seed"]),
    )
    artifact = score_interaction_draw_block(
        *_interaction_tables(data),
        spec,
        draw_start=0,
        draw_end=campaign_contract.B_interaction,
        contract=canonical_execution_contract_from_specs(spec),
        candidate_pair_names=pair_order,
    )
    artifact_dir = shard_dir / "interaction_score_blocks"
    artifact.write_to(artifact_dir)
    selected_by_family = _select_fixed_family_pairs(
        observed_scores=artifact.observed_scores,
        null_scores=artifact.null_scores,
        pair_order=pair_order,
        pair_detectors=artifact.control_snapshot.candidate_pair_detectors,
        family_sizes=campaign_contract.fixed_family_sizes,
        tree_family_alpha=campaign_contract.tree_family_alpha,
        binary_binary_family_alpha=campaign_contract.binary_binary_family_alpha,
    )
    terminal = {
        "operation": "fixed_family_supplement",
        "global_null_replicate_index": int(record["global_null_replicate_index"]),
        "family_sizes": list(campaign_contract.fixed_family_sizes),
        "selected_pairs": selected_by_family,
        "family_partition_method": campaign_contract.family_partition_method,
        "tree_family_alpha": campaign_contract.tree_family_alpha,
        "binary_binary_family_alpha": campaign_contract.binary_binary_family_alpha,
        "artifact_checksum": _score_artifact_payload_sha256(artifact),
        "status": "completed",
    }
    terminal_path = _write_terminal_record(shard_dir, terminal)
    return {
        **terminal,
        "terminal_record": str(terminal_path),
        "scientific_artifacts": _artifact_inventory(shard_dir),
    }


def _applied_specifications(
    record: dict[str, Any], campaign_contract: Any
) -> tuple[Any, Any, Any, Any, Any, Any]:
    from dataclasses import replace

    from rfm_pipeline.manuscript_stages import (
        empirical_null_screening_spec_from_case_study_config,
        final_manuscript_artifacts_spec_from_case_study_config,
        interaction_discovery_spec_from_case_study_config,
        nonlinear_discovery_spec_from_case_study_config,
        output_conditioning_spec_from_case_study_config,
        sparse_selection_stability_spec_from_case_study_config,
    )

    case_study = _load_applied_case_config(record, campaign_contract)
    n_jobs = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    conditioning = output_conditioning_spec_from_case_study_config(case_study)
    screening = replace(
        empirical_null_screening_spec_from_case_study_config(case_study),
        permutation_count_B=campaign_contract.B_screen,
        bh_q_screen=campaign_contract.q_screen,
        random_seed=int(record["seed"]),
        n_jobs=n_jobs,
    )
    interaction = replace(
        interaction_discovery_spec_from_case_study_config(case_study),
        permutation_count_B=campaign_contract.B_interaction,
        random_seed=int(record["seed"]),
        n_tree_estimators=campaign_contract.n_tree_estimators,
        n_jobs=n_jobs,
        selection_method=campaign_contract.method_name,
        selection_alpha=campaign_contract.alpha,
        minimum_selection_draws=campaign_contract.B_interaction,
    )
    nonlinear = replace(
        nonlinear_discovery_spec_from_case_study_config(case_study),
        n_jobs=n_jobs,
    )
    sparse = replace(
        sparse_selection_stability_spec_from_case_study_config(case_study),
        subsample_count=campaign_contract.n_stability_subsamples,
        jaccard_threshold=campaign_contract.stability_jaccard_threshold,
        spearman_threshold=campaign_contract.stability_spearman_threshold,
        n_jobs=n_jobs,
        adaptive_early_stopping_enabled=False,
    )
    final = replace(
        final_manuscript_artifacts_spec_from_case_study_config(case_study),
        bootstrap_count=campaign_contract.bootstrap_draws,
        n_jobs=n_jobs,
    )
    return conditioning, screening, interaction, nonlinear, sparse, final


def _applied_terminal_payload(
    record: dict[str, Any], shard_dir: Path, **values: Any
) -> dict[str, Any]:
    payload = {
        "operation": str(record["operation"]),
        "stage": str(record["stage"]),
        "shard_id": str(record["shard_id"]),
        "status": "completed",
        **values,
    }
    path = _write_terminal_record(shard_dir, payload)
    return {
        **payload,
        "terminal_record": str(path),
        "scientific_artifacts": _artifact_inventory(shard_dir),
    }


def _execute_applied_conditioning(
    record: dict[str, Any], shard_dir: Path, campaign_contract: Any
) -> dict[str, Any]:
    from rfm_pipeline.manuscript_stages import (
        condition_manuscript_outputs,
        write_output_conditioning_artifacts,
    )

    outputs, assignments = _load_applied_train_tables(
        record, names=("outputs", "assignments")
    )
    conditioning, *_ = _applied_specifications(record, campaign_contract)
    result = condition_manuscript_outputs(outputs, assignments, conditioning)
    paths = write_output_conditioning_artifacts(result, shard_dir)
    return _applied_terminal_payload(
        record,
        shard_dir,
        retained_output_count=len(result.retained_output_names),
        culled_output_count=len(result.culled_output_names),
        artifact_paths={name: str(path) for name, path in paths.items()},
    )


def _execute_applied_screening(
    record: dict[str, Any], shard_dir: Path, campaign_contract: Any
) -> dict[str, Any]:
    from rfm_pipeline.manuscript_stages import score_screening_draw_block

    inputs, assignments, catalog = _load_applied_train_tables(
        record, names=("inputs", "assignments", "catalog")
    )
    conditioning = _load_conditioning(record)
    _, screening, *_ = _applied_specifications(record, campaign_contract)
    block = score_screening_draw_block(
        inputs,
        catalog,
        assignments,
        conditioning.pca_scores,
        screening,
        draw_start=int(record["block_start"]),
        draw_end=int(record["block_end"]),
    )
    block.write(shard_dir / "screening_block")
    return _applied_terminal_payload(
        record,
        shard_dir,
        draw_start=block.draw_start,
        draw_end=block.draw_end,
        artifact_hash=block.artifact_hash,
    )


def _execute_applied_interaction(
    record: dict[str, Any], shard_dir: Path, campaign_contract: Any
) -> dict[str, Any]:
    from rfm_pipeline.interaction_contract import (
        canonical_execution_contract_from_specs,
    )
    from rfm_pipeline.manuscript_stages import score_interaction_draw_block

    inputs, assignments, catalog = _load_applied_train_tables(
        record, names=("inputs", "assignments", "catalog")
    )
    conditioning = _load_conditioning(record)
    screening = _load_screening(record)
    _, _, interaction, *_ = _applied_specifications(record, campaign_contract)
    artifact = score_interaction_draw_block(
        inputs,
        catalog,
        assignments,
        conditioning.pca_scores,
        screening.retained_terms,
        interaction,
        draw_start=int(record["block_start"]),
        draw_end=int(record["block_end"]),
        contract=canonical_execution_contract_from_specs(interaction),
    )
    artifact.write_to(shard_dir / "interaction_block")
    return _applied_terminal_payload(
        record,
        shard_dir,
        draw_start=artifact.draw_range_start,
        draw_end=artifact.draw_range_end,
        pair_count=len(artifact.pair_names),
        artifact_checksum=_score_artifact_payload_sha256(artifact),
    )


def _execute_applied_nonlinear(
    record: dict[str, Any], shard_dir: Path, campaign_contract: Any
) -> dict[str, Any]:
    from rfm_pipeline.manuscript_stages import (
        _generate_supported_nonlinear_candidates,
        discover_manuscript_nonlinear_transformations,
    )

    inputs, assignments, catalog = _load_applied_train_tables(
        record, names=("inputs", "assignments", "catalog")
    )
    conditioning = _load_conditioning(record)
    screening = _load_screening(record)
    _, _, _, nonlinear, *_ = _applied_specifications(record, campaign_contract)
    retained_names = screening.retained_terms["feature_name"].astype(str).tolist()
    base_names = tuple(
        dict.fromkeys(
            base
            for _, base, _ in _generate_supported_nonlinear_candidates(
                retained_names,
                inputs,
                transform_library=nonlinear.transform_library,
            )
        )
    )
    start = int(record["block_start"])
    end = min(int(record["block_end"]), len(base_names))
    if start < end:
        result = discover_manuscript_nonlinear_transformations(
            inputs,
            catalog,
            assignments,
            conditioning.pca_scores,
            screening.retained_terms,
            nonlinear,
            checkpoint_dir=shard_dir / "checkpoints",
            active_feature_indices=set(range(start, end)),
        )
        preview_count = len(result.retained_transformations)
    else:
        preview_count = 0
    return _applied_terminal_payload(
        record,
        shard_dir,
        base_feature_count=len(base_names),
        active_start=start,
        active_end=end,
        retained_preview_count=preview_count,
    )


def _sparse_inputs(record: dict[str, Any]) -> tuple[Any, ...]:
    inputs, assignments, catalog = _load_applied_train_tables(
        record, names=("inputs", "assignments", "catalog")
    )
    conditioning = _load_conditioning(record)
    screening = _load_screening(record)
    interactions = _load_interactions(record)
    nonlinear = _load_nonlinear(record)
    return (
        inputs,
        catalog,
        assignments,
        conditioning.pca_scores,
        screening.retained_terms,
        interactions.retained_pairs,
        nonlinear.retained_transformations,
    )


def _execute_applied_sparse_full(
    record: dict[str, Any], shard_dir: Path, campaign_contract: Any
) -> dict[str, Any]:
    from rfm_pipeline.manuscript_stages import fit_sparse_full_selection_artifact

    *_, sparse, _ = _applied_specifications(record, campaign_contract)
    full_fit = fit_sparse_full_selection_artifact(*_sparse_inputs(record), sparse)
    full_fit.write(shard_dir / "sparse_full_fit")
    return _applied_terminal_payload(
        record,
        shard_dir,
        full_fit_hash=full_fit.artifact_hash,
        candidate_count=len(full_fit.candidate_names),
    )


def _execute_applied_sparse_resample(
    record: dict[str, Any], shard_dir: Path, campaign_contract: Any
) -> dict[str, Any]:
    from rfm_pipeline.manuscript_stages import (
        SparseFullFitArtifact,
        score_sparse_stability_resample,
    )

    *_, sparse, _ = _applied_specifications(record, campaign_contract)
    full_fit = SparseFullFitArtifact.read(
        _reducer_root(record, "applied_sparse_full") / "sparse_full_fit"
    )
    start = int(record["block_start"])
    end = int(record["block_end"])
    sparse_inputs = _sparse_inputs(record)
    blocks = []
    for zero_based_id in range(start, end):
        block = score_sparse_stability_resample(
            *sparse_inputs,
            sparse,
            full_fit=full_fit,
            resample_id=zero_based_id + 1,
        )
        block.write(
            shard_dir / "sparse_stability_blocks" / f"resample-{block.resample_id:04d}"
        )
        blocks.append(block)
    return _applied_terminal_payload(
        record,
        shard_dir,
        full_fit_hash=full_fit.artifact_hash,
        block_start=start,
        block_end=end,
        resample_ids=[block.resample_id for block in blocks],
        artifact_hashes=[block.artifact_hash for block in blocks],
    )


_ABLATION_MODELS = (
    "null_mean",
    "main_effects_ols",
    "screened_ols",
    "penalized_ols",
    "final_ols",
)


def _aligned_train_design(
    record: dict[str, Any], feature_names: list[str]
) -> tuple[Any, Any]:
    import pandas as pd

    from rfm_pipeline.manuscript_stages import build_manuscript_feature_design

    inputs, outputs, assignments = _load_applied_train_tables(
        record, names=("inputs", "outputs", "assignments")
    )
    train_ids = assignments.loc[
        assignments["split"].astype(str) == "train", "sample_id"
    ]
    if len(train_ids) != 28500:
        raise ValueError("applied adaptive split does not contain exactly 28,500 rows")
    catalog = pd.DataFrame({"feature_name": feature_names})
    design = build_manuscript_feature_design(inputs, catalog).set_index("sample_id")
    response = outputs.set_index("sample_id")
    ordered_ids = train_ids.tolist()
    return design.loc[ordered_ids, feature_names], response.loc[ordered_ids]


def _execute_applied_terminal_fit(
    record: dict[str, Any], shard_dir: Path, campaign_contract: Any
) -> dict[str, Any]:
    from rfm_pipeline.manuscript_stages import (
        terminal_train_fit_and_freeze,
        write_terminal_train_fit_and_freeze,
    )

    sparse_result = _load_sparse(record)
    feature_names = (
        sparse_result.final_stable_support["feature_name"].astype(str).tolist()
    )
    x_train, y_train = _aligned_train_design(record, feature_names)
    *_, final = _applied_specifications(record, campaign_contract)
    terminal = terminal_train_fit_and_freeze(
        x_train,
        y_train,
        spec=final,
        contract_hash=str(record["config_hash"]),
    )
    write_terminal_train_fit_and_freeze(
        terminal=terminal, output_dir=shard_dir / "terminal"
    )
    return _applied_terminal_payload(
        record,
        shard_dir,
        freeze_hash=terminal.freeze_result.freeze_manifest.freeze_hash,
        prefilter_count=len(terminal.prefilter_feature_names),
        hc3_count=len(terminal.hc3_feature_names),
        final_count=len(terminal.final_feature_names),
    )


def _terminal_feature_names(record: dict[str, Any]) -> list[str]:
    payload = json.loads(
        (
            _reducer_root(record, "applied_terminal_fit")
            / "terminal"
            / "terminal_manifest.json"
        ).read_text(encoding="utf-8")
    )
    return [str(value) for value in payload["final_feature_names"]]


def _execute_applied_ablation_fit(
    record: dict[str, Any], shard_dir: Path, campaign_contract: Any
) -> dict[str, Any]:
    from rfm_pipeline.manuscript_stages import (
        _load_train_fit_and_freeze,
        train_fit_and_freeze,
        train_mean_and_freeze,
        write_train_fit_and_freeze,
    )

    model = str(record["ablation_model"])
    if model not in _ABLATION_MODELS:
        raise ValueError(f"unknown applied ablation model {model!r}")
    outputs, assignments, catalog = _load_applied_train_tables(
        record, names=("outputs", "assignments", "catalog")
    )
    train_ids = assignments.loc[
        assignments["split"].astype(str) == "train", "sample_id"
    ].tolist()
    y_train = outputs.set_index("sample_id").loc[train_ids]
    if model == "final_ols":
        source = _reducer_root(record, "applied_terminal_fit") / "terminal" / "model"
        freeze = _load_train_fit_and_freeze(source)
    elif model == "null_mean":
        freeze = train_mean_and_freeze(
            y_train.to_numpy(dtype=float),
            output_names=y_train.columns.astype(str).tolist(),
            contract_hash=str(record["config_hash"]),
        )
    else:
        screening = _load_screening(record)
        sparse_result = _load_sparse(record)
        if model == "main_effects_ols":
            feature_names = (
                catalog.loc[
                    catalog["feature_type"].astype(str) == "first_order", "feature_name"
                ]
                .astype(str)
                .tolist()
            )
            if len(feature_names) != 160 or len(set(feature_names)) != 160:
                raise ValueError(
                    "main-effects ablation requires exactly 160 unique inputs"
                )
        elif model == "screened_ols":
            feature_names = (
                screening.retained_terms["feature_name"].astype(str).tolist()
            )
        else:
            feature_names = (
                sparse_result.final_stable_support["feature_name"].astype(str).tolist()
            )
        if feature_names:
            x_train, y_train = _aligned_train_design(record, feature_names)
            freeze = train_fit_and_freeze(
                x_train.to_numpy(dtype=float),
                y_train.to_numpy(dtype=float),
                feature_names=feature_names,
                output_names=y_train.columns.astype(str).tolist(),
                contract_hash=str(record["config_hash"]),
            )
        else:
            freeze = train_mean_and_freeze(
                y_train.to_numpy(dtype=float),
                output_names=y_train.columns.astype(str).tolist(),
                contract_hash=str(record["config_hash"]),
            )
    model_dir = shard_dir / "model"
    write_train_fit_and_freeze(freeze_result=freeze, output_dir=model_dir)
    return _applied_terminal_payload(
        record,
        shard_dir,
        ablation_model=model,
        freeze_hash=freeze.freeze_manifest.freeze_hash,
        feature_count=len(freeze.freeze_manifest.feature_names),
        output_count=len(freeze.freeze_manifest.output_names),
    )


def _execute_applied_holdout_authorize(
    record: dict[str, Any], shard_dir: Path, campaign_contract: Any
) -> dict[str, Any]:
    del campaign_contract
    from rfm_pipeline.manuscript_stages import _load_train_fit_and_freeze

    root, manifest = _verify_applied_data_manifest(record)
    driver = _load_driver(record)
    _verify_phase_authorization(
        driver, record, contract_hash=str(record["config_hash"])
    )
    freeze_hashes = {}
    models_root = _reducer_root(record, "applied_ablation_fit") / "models"
    for model in _ABLATION_MODELS:
        freeze = _load_train_fit_and_freeze(models_root / model)
        freeze_hashes[model] = freeze.freeze_manifest.freeze_hash
    terminal_freeze = _load_train_fit_and_freeze(
        _reducer_root(record, "applied_terminal_fit") / "terminal" / "model"
    )
    if freeze_hashes["final_ols"] != terminal_freeze.freeze_manifest.freeze_hash:
        raise ValueError("final ablation freeze differs from the terminal model freeze")
    holdout_y = root / "sealed_holdout" / "Y.parquet"
    expected = manifest["generated_sha256"]["sealed_holdout/Y.parquet"]
    try:
        os.chmod(holdout_y, 0o400)
        actual = hashlib.sha256(holdout_y.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                "unsealed holdout response differs from the prepared manifest"
            )
        authorization = {
            "schema_version": 1,
            "status": "completed",
            "operation": "applied_holdout_authorize",
            "contract_hash": str(record["config_hash"]),
            "dataset_manifest_sha256": str(record["applied_data_manifest_sha256"]),
            "model_freeze_hashes": freeze_hashes,
            "holdout_response_sha256": actual,
            "authorized_at_unix_ns": time.time_ns(),
        }
        path = shard_dir / "holdout_access_authorization.json"
        path.write_text(
            json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        os.chmod(holdout_y, 0o000)
        raise
    return _applied_terminal_payload(
        record,
        shard_dir,
        holdout_access_authorization=str(path),
        model_freeze_hashes=freeze_hashes,
    )


def _load_holdout_authorization(record: dict[str, Any]) -> dict[str, Any]:
    path = (
        _reducer_root(record, "applied_holdout_authorize")
        / "holdout_access_authorization.json"
    )
    if not path.is_file():
        raise ValueError("holdout prediction lacks a model-freeze access authorization")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "completed"
        or payload.get("contract_hash") != record.get("config_hash")
        or payload.get("dataset_manifest_sha256")
        != record.get("applied_data_manifest_sha256")
    ):
        raise ValueError("holdout access authorization identity differs")
    return payload


def _execute_applied_holdout_predict(
    record: dict[str, Any], shard_dir: Path, campaign_contract: Any
) -> dict[str, Any]:
    del campaign_contract
    import pandas as pd
    import pyarrow.parquet as pq

    from rfm_pipeline.manuscript_stages import (
        FrozenPredictionMatrices,
        _g11_eligibility_ledger_from_statistics,
        _load_train_fit_and_freeze,
        build_manuscript_feature_design,
        write_frozen_prediction_matrices,
    )

    authorization = _load_holdout_authorization(record)
    root, manifest = _verify_applied_data_manifest(record)
    x_path = root / "sealed_holdout" / "X.parquet"
    y_path = root / "sealed_holdout" / "Y.parquet"
    generated = manifest["generated_sha256"]
    if (
        hashlib.sha256(x_path.read_bytes()).hexdigest()
        != generated["sealed_holdout/X.parquet"]
    ):
        raise ValueError("holdout predictor hash changed after preparation")
    if (
        hashlib.sha256(y_path.read_bytes()).hexdigest()
        != generated["sealed_holdout/Y.parquet"]
    ):
        raise ValueError("holdout response hash changed after authorization")
    if (
        authorization.get("holdout_response_sha256")
        != generated["sealed_holdout/Y.parquet"]
    ):
        raise ValueError(
            "holdout authorization does not bind the prepared response bytes"
        )
    holdout_x = pd.read_parquet(x_path)
    output_names = [
        name for name in pq.ParquetFile(y_path).schema.names if name != "sample_id"
    ]
    start = int(record["block_start"])
    end = int(record["block_end"])
    selected_outputs = output_names[start:end]
    holdout_y = pd.read_parquet(y_path, columns=["sample_id", *selected_outputs])
    if len(holdout_x) != 1500 or len(holdout_y) != 1500:
        raise ValueError("holdout prediction requires exactly 1,500 frozen rows")
    if holdout_x["sample_id"].tolist() != holdout_y["sample_id"].tolist():
        raise ValueError("holdout X/Y row identities differ")
    ids = tuple(holdout_y["sample_id"].astype(str))
    strata = tuple(
        f"{int(left)}_{int(right)}"
        for left, right in zip(
            holdout_x["FM.Use Agnostic FS Conversion"],
            holdout_x["OI.Use AEO Reference Oil"],
            strict=True,
        )
    )
    models_root = _reducer_root(record, "applied_ablation_fit") / "models"
    model_hashes = {}
    for model in _ABLATION_MODELS:
        freeze = _load_train_fit_and_freeze(models_root / model)
        if (
            freeze.freeze_manifest.freeze_hash
            != authorization["model_freeze_hashes"][model]
        ):
            raise ValueError(f"holdout model freeze changed for {model}")
        feature_names = list(freeze.freeze_manifest.feature_names)
        if feature_names:
            design = build_manuscript_feature_design(
                holdout_x,
                pd.DataFrame({"feature_name": feature_names}),
            )
            values = design[feature_names].to_numpy(dtype=float)
            standardized = (values - freeze.x_means) / freeze.x_scales
            prediction = (
                standardized @ freeze.coef[:, start:end] + freeze.intercept[start:end]
            )
        else:
            prediction = np.tile(freeze.intercept[start:end], (len(holdout_x), 1))
        manifest_freeze = freeze.freeze_manifest
        ref_min = np.asarray(manifest_freeze.y_train_min[start:end], dtype=float)
        ref_max = np.asarray(manifest_freeze.y_train_max[start:end], dtype=float)
        ref_mean = np.asarray(manifest_freeze.y_train_mean[start:end], dtype=float)
        ref_variance = np.asarray(
            manifest_freeze.y_train_variance[start:end], dtype=float
        )
        frozen = FrozenPredictionMatrices(
            freeze_hash=manifest_freeze.freeze_hash,
            y_holdout=holdout_y[selected_outputs].to_numpy(dtype=float),
            y_pred=prediction,
            y_train_min=ref_min,
            y_train_max=ref_max,
            y_train_mean=ref_mean,
            y_train_variance=ref_variance,
            output_names=tuple(selected_outputs),
            truth_ids=ids,
            prediction_ids=ids,
            strata=strata,
            eligibility_ledger=_g11_eligibility_ledger_from_statistics(
                output_ids=selected_outputs,
                ref_min=ref_min,
                ref_max=ref_max,
                ref_mean=ref_mean,
                ref_variance=ref_variance,
            ),
        )
        model_dir = shard_dir / "models" / model
        write_frozen_prediction_matrices(frozen=frozen, output_dir=model_dir)
        model_hashes[model] = frozen.freeze_hash
    return _applied_terminal_payload(
        record,
        shard_dir,
        output_start=start,
        output_end=end,
        model_freeze_hashes=model_hashes,
    )


def _execute_applied_eligibility(
    record: dict[str, Any], shard_dir: Path, campaign_contract: Any
) -> dict[str, Any]:
    del campaign_contract
    from rfm_pipeline.manuscript_stages import (
        read_frozen_prediction_matrices,
        validate_g11_production_eligibility_ledger,
    )

    frozen = read_frozen_prediction_matrices(
        _reducer_root(record, "applied_holdout_predict") / "models" / "final_ols"
    )
    accounting = validate_g11_production_eligibility_ledger(
        frozen.eligibility_ledger,
        expected_accounting={"total": 23495, "eligible": 9954, "excluded": 13541},
    )
    path = shard_dir / "output_eligibility_ledger.csv"
    frozen.eligibility_ledger.to_csv(path, index=False)
    return _applied_terminal_payload(
        record,
        shard_dir,
        freeze_hash=frozen.freeze_hash,
        accounting=accounting,
        eligibility_ledger=str(path),
    )


def _execute_applied_bootstrap(
    record: dict[str, Any], shard_dir: Path, campaign_contract: Any
) -> dict[str, Any]:
    from rfm_pipeline.manuscript_stages import (
        bootstrap_metric_block,
        read_frozen_prediction_matrices,
    )

    prediction_root = _reducer_root(record, "applied_holdout_predict") / "models"
    draw_start = int(record["block_start"])
    draw_end = int(record["block_end"])
    summaries = {}
    for model in _ABLATION_MODELS:
        frozen = read_frozen_prediction_matrices(prediction_root / model)
        block = bootstrap_metric_block(
            frozen,
            draw_start=draw_start,
            draw_end=draw_end,
            random_seed=int(record["seed"]),
        )
        model_dir = shard_dir / "models" / model
        model_dir.mkdir(parents=True)
        arrays_path = model_dir / "bootstrap_block.npz"
        np.savez_compressed(
            arrays_path,
            per_draw_macro_nrmse=block.per_draw_macro_nrmse,
            draw_ids=block.draw_ids,
        )
        metadata = {
            "freeze_hash": block.freeze_hash,
            "schedule_hash": block.schedule_hash,
            "draw_start": block.draw_start,
            "draw_end": block.draw_end,
            "eligible_count": block.eligible_count,
            "stratum_counts": list(block.stratum_counts),
            "npz_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        }
        (model_dir / "bootstrap_block.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summaries[model] = {
            "freeze_hash": block.freeze_hash,
            "schedule_hash": block.schedule_hash,
        }
    return _applied_terminal_payload(
        record,
        shard_dir,
        draw_start=draw_start,
        draw_end=draw_end,
        models=summaries,
    )


def _execute_applied_work_unit(
    record: dict[str, Any], shard_dir: Path, campaign_contract: Any
) -> dict[str, Any]:
    operations = {
        "applied_conditioning": _execute_applied_conditioning,
        "applied_screening": _execute_applied_screening,
        "applied_interaction": _execute_applied_interaction,
        "applied_nonlinear": _execute_applied_nonlinear,
        "applied_sparse_full": _execute_applied_sparse_full,
        "applied_sparse_resample": _execute_applied_sparse_resample,
        "applied_terminal_fit": _execute_applied_terminal_fit,
        "applied_ablation_fit": _execute_applied_ablation_fit,
        "applied_holdout_authorize": _execute_applied_holdout_authorize,
        "applied_holdout_predict": _execute_applied_holdout_predict,
        "applied_eligibility": _execute_applied_eligibility,
        "applied_bootstrap": _execute_applied_bootstrap,
    }
    operation = str(record["operation"])
    handler = operations.get(operation)
    if handler is None:
        raise ValueError(f"unknown applied operation {operation!r}")
    return handler(record, shard_dir, campaign_contract)


def _discover_nonlinear_allow_empty_family(
    discover: Any,
    input_matrix: Any,
    feature_catalog: Any,
    holdout_assignments: Any,
    pca_scores: Any,
    retained_terms: Any,
    spec: Any,
    **kwargs: Any,
) -> Any:
    """Return a terminal empty result when a valid screen has no nonlinear family."""
    empty_family_error: ValueError | None = None
    try:
        return discover(
            input_matrix,
            feature_catalog,
            holdout_assignments,
            pca_scores,
            retained_terms,
            spec,
            **kwargs,
        )
    except ValueError as exc:
        if (
            str(exc)
            != "feature_catalog does not contain supported nonlinear candidates."
        ):
            raise
        empty_family_error = exc

    import pandas as pd
    from rfm_pipeline.manuscript_stages import (
        NonlinearDiscoveryResult,
        _align_table_by_sample_id,
        _build_nonlinear_discovery_summary,
        _build_nonlinear_provenance,
        _component_columns,
        _generate_supported_nonlinear_candidates,
        _retained_first_order_term_names,
        _standardize_for_screening,
        _train_sample_ids,
    )

    candidates = _generate_supported_nonlinear_candidates(
        _retained_first_order_term_names(retained_terms, input_matrix),
        input_matrix,
        transform_library=spec.transform_library,
    )
    if candidates:
        raise ValueError(
            "nonlinear discovery raised the empty-family terminal error for a non-empty family"
        ) from empty_family_error
    component_names = _component_columns(pca_scores)
    train_ids = _train_sample_ids(holdout_assignments)
    y_train = _align_table_by_sample_id(
        pca_scores, train_ids, component_names, "PCA scores"
    )
    if len(y_train) < 3:
        raise ValueError("Nonlinear discovery requires at least three training rows.")
    _, component_active = _standardize_for_screening(y_train)
    if not component_active.any():
        raise ValueError("All retained PCA components have zero training variance.")

    score_columns = [
        "feature_name",
        "base_feature",
        "transformation_family",
        "curvature_score",
        "gam_p_value",
        "best_component",
        "replacement_training_rmse",
        "active_transform",
        "empirical_null_retained",
        "retained",
        "curvature_rule",
        "replacement_selection_rule",
    ]
    empty_scores = pd.DataFrame(columns=score_columns)
    summary = _build_nonlinear_discovery_summary(
        n_training_rows=len(y_train),
        n_candidate_transformations=0,
        n_active_transformations=0,
        n_empirical_null_retained_transformations=0,
        n_retained_transformations=0,
        n_components=len(component_names),
        max_curvature_score=0.0,
        spec=spec,
    )
    summary.insert(1, "status", "empty_candidate_family")
    return NonlinearDiscoveryResult(
        transformation_scores=empty_scores,
        component_transformation_scores=pd.DataFrame(
            columns=["feature_name", "component", "smooth_edf", "gam_p_value"]
        ),
        retained_transformations=empty_scores.copy(),
        provenance=_build_nonlinear_provenance(spec),
        summary=summary,
    )


def _run_pipeline_allowing_empty_nonlinear_family(
    driver: ModuleType, *args: Any, **kwargs: Any
) -> Any:
    """Run one recovery replicate with the reviewed empty-family terminal rule."""
    import rfm_pipeline.recovery_study as recovery_study

    original = recovery_study.discover_manuscript_nonlinear_transformations

    def _discover(*call_args: Any, **call_kwargs: Any) -> Any:
        return _discover_nonlinear_allow_empty_family(
            original, *call_args, **call_kwargs
        )

    recovery_study.discover_manuscript_nonlinear_transformations = _discover
    try:
        return driver.run_pipeline(*args, **kwargs)
    finally:
        recovery_study.discover_manuscript_nonlinear_transformations = original


def execute_scientific_work_unit(
    record: dict[str, Any], shard_dir: Path
) -> dict[str, Any]:
    """Execute exactly one manifest-bound Gate-B/Gate-C replicate."""
    from rfm_pipeline.campaign_contract import load_contract

    operation = str(record.get("operation", ""))
    if operation not in {
        "resolution",
        "gate_b",
        "fixed_family_supplement",
        "recovery",
        "applied_conditioning",
        "applied_screening",
        "applied_interaction",
        "applied_nonlinear",
        "applied_sparse_full",
        "applied_sparse_resample",
        "applied_terminal_fit",
        "applied_ablation_fit",
        "applied_holdout_authorize",
        "applied_holdout_predict",
        "applied_eligibility",
        "applied_bootstrap",
    }:
        raise ValueError(f"BSM campaign adapter does not implement {operation!r}")
    campaign_contract_path = record.get("contract_config_path")
    if not campaign_contract_path:
        raise ValueError("manifest record is missing contract_config_path")
    campaign_contract, contract_hash = load_contract(Path(str(campaign_contract_path)))
    if record.get("config_hash") != contract_hash:
        raise ValueError("BSM adapter campaign hash differs from its manifest record")
    resource_freeze_sha256 = str(record.get("resource_freeze_sha256", ""))
    if len(resource_freeze_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in resource_freeze_sha256
    ):
        raise ValueError("BSM adapter requires a content-bound resource freeze")
    if (
        operation != "resolution"
        and campaign_contract.resolution_decision_sha256 == "PENDING"
    ):
        raise ValueError(
            "confirmatory operation uses a provisional pre-resolution contract"
        )
    dgp_contract_path = _require_hashed_file(
        record,
        path_field="bsm_dgp_contract_path",
        hash_field="bsm_dgp_contract_sha256",
    )
    driver = _load_driver(record)
    dgp_contract = driver.load_bsm_dgp_contract(dgp_contract_path)
    design = driver.load_bsm_input_design(contract=dgp_contract)
    if operation.startswith("applied_"):
        _verify_phase_authorization(driver, record, contract_hash=contract_hash)
        return _execute_applied_work_unit(record, shard_dir, campaign_contract)
    if operation in {"resolution", "fixed_family_supplement"}:
        _verify_phase_authorization(driver, record, contract_hash=contract_hash)
        if operation == "resolution":
            return _execute_resolution(
                record=record,
                shard_dir=shard_dir,
                campaign_contract=campaign_contract,
                driver=driver,
                dgp_contract=dgp_contract,
                design=design,
            )
        return _execute_fixed_family(
            record=record,
            shard_dir=shard_dir,
            campaign_contract=campaign_contract,
            driver=driver,
            dgp_contract=dgp_contract,
            design=design,
        )
    _verify_phase_authorization(driver, record, contract_hash=contract_hash)
    started = time.perf_counter()
    data = driver.generate_campaign_dataset(
        campaign_contract=campaign_contract,
        bsm_contract=dgp_contract,
        design=design,
        scenario_id=str(record["scenario_id"]),
        replicate_index=int(record["replicate_index"]),
    )
    scenario_kind = str(record.get("scenario_kind", ""))
    include_recovery_comparators = _requires_recovery_comparators(scenario_kind)
    expected_comparators = (
        campaign_contract.recovery_comparators if include_recovery_comparators else ()
    )
    if tuple(record.get("recovery_comparators", ())) != expected_comparators:
        raise ValueError(
            "manifest recovery comparator population differs from the contract"
        )
    result = _run_pipeline_allowing_empty_nonlinear_family(
        driver,
        data,
        dgp_contract,
        campaign_contract_path=Path(str(campaign_contract_path)),
        artifact_dir=shard_dir / "pipeline",
        authorization_manifest=Path(str(record["execution_authorization_path"])),
        phase=str(record["phase"]),
        seed=int(record["seed"]),
        include_recovery_comparators=include_recovery_comparators,
    )
    truth_pairs = _canonical_support(
        [
            _truth_support_name(term, design)
            for term in data.truth.terms
            if term.term_id.kind.endswith("_interaction")
        ],
        field="truth_interaction_ids",
    )
    retained_pairs = _canonical_support(
        list(result.interaction_retained_set), field="retained_interaction_ids"
    )
    in_library_truth = _canonical_support(
        [
            _truth_support_name(term, design)
            for term in data.truth.terms
            if term.in_library
        ],
        field="in_library_truth_support",
    )
    selected_support = _canonical_support(
        list(result.final_selected_support), field="final_selected_support"
    )
    transform_suffixes = ("_sq", "_log1p", "_inv", "_sqrt")
    truth_families = {
        "main": {
            name
            for name in in_library_truth
            if ":" not in name and not name.endswith(transform_suffixes)
        },
        "interaction": {name for name in in_library_truth if ":" in name},
        "transformation": {
            name for name in in_library_truth if name.endswith(transform_suffixes)
        },
    }
    selected_families = {
        "main": {
            name
            for name in selected_support
            if ":" not in name and not name.endswith(transform_suffixes)
        },
        "interaction": {name for name in selected_support if ":" in name},
        "transformation": {
            name for name in selected_support if name.endswith(transform_suffixes)
        },
    }
    recovery_metrics = {
        family: _support_metrics(truth_families[family], selected_families[family])
        for family in truth_families
    }
    recovery_metrics["whole"] = _support_metrics(in_library_truth, selected_support)
    comparator_predictions = result.comparator_predictions or {}
    if set(comparator_predictions) != set(expected_comparators):
        raise ValueError(
            "recovery comparator result does not cover the frozen comparator set"
        )
    prediction_metrics = (
        _prediction_metrics(data, comparator_predictions)
        if include_recovery_comparators
        else {}
    )
    terminal_record = {
        "operation": str(record["operation"]),
        "scenario": str(record["scenario_id"]),
        "replicate_index": int(record["replicate_index"]),
        "seed": int(record["seed"]),
        "contract_hash": contract_hash,
        "source_hash": str(record["source_hash"]),
        "schedule_hash": str(record["schedule_hash"]),
        "screened_count": len(result.screening_retained_set),
        "pair_family_count": int(result.interaction_candidate_count),
        "truth_pairs": len(truth_pairs),
        "retained_pairs": len(retained_pairs),
        "false_pair_count": len(retained_pairs.difference(truth_pairs)),
        "terminal_stage": result.terminal_status,
        "status": "completed",
        "exception": "",
        "runtime_s": time.perf_counter() - started,
        "peak_memory_mb": 0.0,
        "truth_interaction_ids": sorted(truth_pairs),
        "retained_interaction_ids": sorted(retained_pairs),
        "planted_interaction_discovered": truth_pairs <= retained_pairs,
        "in_library_truth_support": sorted(in_library_truth),
        "out_of_library_truth_support": sorted(
            _truth_support_name(term, design)
            for term in data.truth.terms
            if not term.in_library
        ),
        "final_selected_support": sorted(selected_support),
        "recovery_metrics": recovery_metrics,
        "macro_nrmse_by_method": prediction_metrics,
        "metric_output_count": int(data.Y_eval.shape[1]),
        "model_freeze_hash": result.model_freeze_hash,
        "comparator_schedule_sha256": result.comparator_schedule_sha256,
    }
    terminal_path = shard_dir / "terminal_record.json"
    terminal_path.write_text(
        json.dumps(terminal_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    truth_path = shard_dir / "truth_ledger.json"
    truth_path.write_text(
        json.dumps(data.truth_ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "operation": str(record["operation"]),
        "status": "completed",
        "terminal_record": str(terminal_path),
        "truth_ledger": str(truth_path),
        "terminal_status": result.terminal_status,
        "screened_count": len(result.screening_retained_set),
        "pair_family_count": result.interaction_candidate_count,
        "retained_pair_count": len(result.interaction_retained_set),
        "comparator_names": sorted(result.comparator_predictions or {}),
        "prediction_shape": list(np.asarray(result.eval_predictions).shape),
        "scientific_artifacts": _artifact_inventory(shard_dir),
    }
