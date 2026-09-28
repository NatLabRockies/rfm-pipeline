"""Compile final G11 reducer outputs into publication tables, CSVs, and LaTeX.

The compiler performs no model fitting and never reads raw training inputs.  It
accepts only a completed confirmatory campaign package and immutable reducer
outputs, then creates a new checksummed publication directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_MODELS = (
    "null_mean",
    "main_effects_ols",
    "screened_ols",
    "penalized_ols",
    "final_ols",
)
_EXPECTED_TOTAL_OUTPUTS = 23_495
_EXPECTED_ELIGIBLE_OUTPUTS = 9_954
_EXPECTED_EXCLUDED_OUTPUTS = 13_541


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required publication input is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"publication JSON input is not an object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_verified(source: Path, destination: Path, *, expected_sha256: str) -> None:
    if not source.is_file() or _sha256(source) != expected_sha256:
        raise ValueError(
            f"publication provenance input is absent or hash-mismatched: {source}"
        )
    shutil.copy2(source, destination)


def _stage_reducer(results_root: Path, stage: str) -> Path:
    return results_root / "stages" / stage / "reducer"


def _stage_summary(results_root: Path, stage: str) -> dict[str, Any]:
    payload = _read_json(
        _stage_reducer(results_root, stage) / "scientific_stage_summary.json"
    )
    if payload.get("operation") != stage or payload.get("status") != "completed":
        raise ValueError(f"stage {stage} does not have a completed scientific summary")
    return payload


def _verify_completed_reducer(
    *,
    package_summary: dict[str, Any],
    results_root: Path,
    stage: str,
    contract_hash: str,
) -> None:
    """Bind every consumed reducer directory to the confirmatory package."""
    stage_rows = package_summary.get("stages")
    if not isinstance(stage_rows, list):
        raise ValueError("confirmatory package does not contain stage identities")
    matches = [row for row in stage_rows if row.get("name") == stage]
    if len(matches) != 1:
        raise ValueError(
            f"confirmatory package does not uniquely identify stage {stage}"
        )
    packaged = matches[0]
    reducer = _stage_reducer(results_root, stage)
    reduced_path = reducer / "reduced_result.json"
    marker = _read_json(reducer / "_SUCCESS.json")
    reduced = _read_json(reduced_path)
    if (
        marker.get("status") != "completed"
        or marker.get("stage") != stage
        or marker.get("output_hash") != packaged.get("reducer_output_hash")
    ):
        raise ValueError(
            f"stage {stage} reducer output identity differs from the package"
        )
    if marker.get("artifact_sha256") != _sha256(reduced_path):
        raise ValueError(f"stage {stage} reducer success checksum differs")
    if (
        reduced.get("status") != "completed"
        or reduced.get("stage") != stage
        or reduced.get("contract_hash") != contract_hash
        or int(reduced.get("records", -1)) != int(packaged.get("job_count", -2))
    ):
        raise ValueError(
            f"stage {stage} reduced result differs from the package contract"
        )
    scientific = reduced.get("scientific_reduction")
    artifacts = (
        scientific.get("scientific_artifacts") if isinstance(scientific, dict) else None
    )
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(
            f"stage {stage} reduced result has no scientific artifact inventory"
        )
    reducer_root = reducer.resolve()
    seen: set[Path] = set()
    for item in artifacts:
        path = Path(str(item.get("path", ""))).resolve()
        try:
            path.relative_to(reducer_root)
        except ValueError as exc:
            raise ValueError(
                f"stage {stage} scientific artifact escapes its reducer"
            ) from exc
        if path in seen or not path.is_file() or item.get("sha256") != _sha256(path):
            raise ValueError(f"stage {stage} scientific artifact bytes differ")
        seen.add(path)


def _strict_bool_series(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    values = series.astype("string")
    if values.isna().any() or not values.isin(("True", "False")).all():
        raise ValueError(f"{name} must contain only explicit True/False values")
    return values.map({"True": True, "False": False}).astype(bool)


def _feature_type(name: str) -> str:
    from rfm_pipeline.manuscript_stages import _legacy_feature_type_label

    return str(_legacy_feature_type_label(name))


def _canonical_interaction_id(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("interaction identifier must be a string")
    parts = value.split(":")
    if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
        raise ValueError(f"invalid interaction identifier {value!r}")
    return ":".join(sorted(parts))


def _canonical_support(record: dict[str, Any], field: str) -> set[str]:
    values = record.get(field)
    if not isinstance(values, list):
        raise ValueError(f"recovery terminal record lacks {field}")
    canonical: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} contains an invalid support identifier")
        normalized = _canonical_interaction_id(value) if ":" in value else value
        if normalized in canonical:
            raise ValueError(f"{field} contains duplicate semantic support")
        canonical.add(normalized)
    return canonical


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


def _recompute_recovery_metrics(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Derive support metrics from raw IDs, ignoring stale cached summaries."""
    truth = _canonical_support(record, "in_library_truth_support")
    selected = _canonical_support(record, "final_selected_support")
    transform_suffixes = ("_sq", "_log1p", "_inv", "_sqrt")

    def family(values: set[str], name: str) -> set[str]:
        if name == "interaction":
            return {value for value in values if ":" in value}
        if name == "transformation":
            return {value for value in values if value.endswith(transform_suffixes)}
        return {
            value
            for value in values
            if ":" not in value and not value.endswith(transform_suffixes)
        }

    metrics = {
        name: _support_metrics(family(truth, name), family(selected, name))
        for name in ("main", "interaction", "transformation")
    }
    metrics["whole"] = _support_metrics(truth, selected)
    return metrics


def _write_wide_matrix(
    path: Path,
    *,
    feature_names: list[str],
    output_names: list[str],
    values: np.ndarray,
) -> None:
    if values.shape != (len(feature_names), len(output_names)):
        raise ValueError(f"coefficient matrix shape differs for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_name", *output_names])
        for name, row in zip(feature_names, values, strict=True):
            writer.writerow([name, *row.tolist()])


def _write_manuscript_macros(path: Path, values: dict[str, Any]) -> None:
    lines = ["% Generated from one immutable G11 campaign; do not hand-edit."]
    for name, value in values.items():
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tex_escape(value: str) -> str:
    translations = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(translations.get(character, character) for character in value)


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "publication_artifact_manifest.json":
            continue
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows


def _validate_public_provenance(root: Path) -> None:
    forbidden = re.compile(
        r"(?:/Users/|/(?:kfs[0-9]*|scratch|projects)/|dhetting|@nlr\.gov)",
        re.IGNORECASE,
    )
    text_suffixes = {".json", ".jsonl", ".toml", ".yml", ".yaml", ".py", ".md"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in text_suffixes:
            if forbidden.search(path.read_text(encoding="utf-8", errors="replace")):
                raise ValueError(
                    f"public provenance contains an internal path or identity: {path.name}"
                )


def _load_model_feature_counts(results_root: Path) -> dict[str, int]:
    root = _stage_reducer(results_root, "applied_ablation_fit") / "models"
    counts = {}
    for model in _MODELS:
        manifest = _read_json(root / model / "freeze_manifest.json")
        names = manifest.get("feature_names")
        if not isinstance(names, list) or len(names) != len(set(map(str, names))):
            raise ValueError(f"ablation model {model} has invalid feature identity")
        counts[model] = len(names)
    if counts["main_effects_ols"] != 160:
        raise ValueError("main-effects ablation does not contain all 160 inputs")
    return counts


def _build_recovery_tables(
    results_root: Path,
    tables_dir: Path,
    manuscript_dir: Path,
    contract_hash: str,
) -> dict[str, int]:
    gate_b = _read_json(_stage_reducer(results_root, "gate_b") / "gate_b_decision.json")
    fixed = _read_json(
        _stage_reducer(results_root, "fixed_family_supplement")
        / "fixed_family_decision.json"
    )
    gate_c = _read_json(
        _stage_reducer(results_root, "recovery") / "gate_c_stress_ledger_summary.json"
    )
    for label, payload in (("gate_b", gate_b), ("fixed_family", fixed)):
        if (
            payload.get("status") != "completed"
            or payload.get("decision") != "PASS"
            or payload.get("contract_hash") != contract_hash
        ):
            raise ValueError(f"{label} did not pass under the final contract")
    if (
        gate_c.get("status") != "completed"
        or gate_c.get("acceptance_status") != "READY_FOR_INDEPENDENT_REVIEW"
        or gate_c.get("contract_hash") != contract_hash
    ):
        raise ValueError("Gate C is incomplete or belongs to another contract")

    calibration_rows = []
    for scenario, values in gate_b.get("scenarios", {}).items():
        if "fwer" in values and not bool(values.get("passes_calibration")):
            raise ValueError(f"Gate B calibration row does not pass: {scenario}")
        if "power" in values and not bool(values.get("passes_power")):
            raise ValueError(f"Gate B power row does not pass: {scenario}")
        calibration_rows.append({"source": "gate_b", "scenario": scenario, **values})
    for family, values in fixed.get("families", {}).items():
        if not bool(values.get("passes_calibration")):
            raise ValueError(f"fixed-family calibration row does not pass: {family}")
        calibration_rows.append(
            {
                "source": "fixed_family_supplement",
                "scenario": f"family_{family}",
                **values,
            }
        )
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(tables_dir / "fwer_calibration.csv", index=False)
    fwer = calibration.loc[calibration["fwer"].notna()].copy()
    if fwer.empty:
        raise ValueError("final recovery bundle contains no FWER calibration rows")
    fwer_rows = []
    for row in fwer.to_dict(orient="records"):
        label = str(row["scenario"]).replace("_", " ").title()
        if row["source"] == "fixed_family_supplement":
            label = f"Fixed family: {label.removeprefix('Family ')} pairs"
        fwer_rows.append(
            f"{_tex_escape(label)} & {int(row['denominator']):,} & "
            f"{float(row['fwer']):.3f} & {float(row['one_sided_wilson_upper']):.3f} "
            "& Pass"
        )
    (manuscript_dir / "recovery_fwer_rows.tex").write_text(
        "% Generated from the final Gate-B decisions; do not hand-edit.\n"
        + " \\\\\n".join(fwer_rows)
        + "\n",
        encoding="utf-8",
    )

    terminal_paths_by_stage = {
        stage: sorted(
            (results_root / "stages" / stage / "results").glob("*/terminal_record.json")
        )
        for stage in ("gate_b", "fixed_family_supplement", "recovery")
    }
    for stage, paths in terminal_paths_by_stage.items():
        reduced = _read_json(
            _stage_reducer(results_root, stage) / "reduced_result.json"
        )
        if len(paths) != int(reduced.get("records", -1)):
            raise ValueError(f"stage {stage} terminal-record coverage differs")
        expected_result_hashes = sorted(map(str, reduced.get("artifact_hashes", [])))
        observed_result_hashes = sorted(
            _sha256(path.parent / "result.json") for path in paths
        )
        if observed_result_hashes != expected_result_hashes:
            raise ValueError(
                f"stage {stage} worker result bytes differ from its reducer"
            )
    terminal_paths = [
        path
        for stage in ("gate_b", "fixed_family_supplement", "recovery")
        for path in terminal_paths_by_stage[stage]
    ]
    records = []
    for path in terminal_paths:
        result = _read_json(path.parent / "result.json")
        record = _read_json(path)
        operation = str(record.get("operation", ""))
        artifacts = result.get("scientific_artifacts")
        artifact_is_bound = isinstance(artifacts, list) and any(
            Path(str(item.get("path", ""))).resolve() == path.resolve()
            and item.get("sha256") == _sha256(path)
            for item in artifacts
        )
        if (
            result.get("status") != "completed"
            or result.get("operation") != operation
            or Path(str(result.get("terminal_record", ""))).resolve() != path.resolve()
            or not artifact_is_bound
        ):
            raise ValueError(
                "recovery terminal record is not bound to its worker result"
            )
        if (
            operation in {"gate_b", "recovery"}
            and record.get("contract_hash") != contract_hash
        ):
            raise ValueError("recovery terminal record has a stale contract")
        if operation == "fixed_family_supplement" and set(
            map(str, record.get("family_sizes", []))
        ) != set(map(str, fixed.get("families", {}))):
            raise ValueError("fixed-family terminal record differs from its decision")
        records.append(record)
    if not records:
        raise ValueError(
            "recovery artifact compiler found no terminal replicate records"
        )
    rows = []
    for record in records:
        if record.get("status") != "completed":
            raise ValueError("recovery terminal record is incomplete")
        if record.get("recovery_metrics"):
            record["recovery_metrics"] = _recompute_recovery_metrics(record)
        rows.append(
            {
                "operation": record.get("operation"),
                "scenario": record.get("scenario"),
                "replicate_index": record.get("replicate_index"),
                "seed": record.get("seed"),
                "recovery_metrics_json": json.dumps(
                    record.get("recovery_metrics", {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "macro_nrmse_by_method_json": json.dumps(
                    record.get("macro_nrmse_by_method", {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    pd.DataFrame(rows).to_csv(tables_dir / "recovery_replicates.csv", index=False)
    scientific_records = [
        record for record in records if record.get("recovery_metrics")
    ]
    for record in scientific_records:
        methods = record.get("macro_nrmse_by_method", {})
        if methods and "proposed_terminal_workflow" not in methods:
            raise ValueError(
                "recovery comparator metrics lack the frozen proposed-terminal method"
            )
    scenario_rows = []
    for scenario in sorted({str(record["scenario"]) for record in scientific_records}):
        subset = [
            record
            for record in scientific_records
            if str(record["scenario"]) == scenario
        ]
        whole = [record["recovery_metrics"]["whole"] for record in subset]
        interactions = [record["recovery_metrics"]["interaction"] for record in subset]
        proposed = [
            float(record["macro_nrmse_by_method"]["proposed_terminal_workflow"])
            for record in subset
            if "proposed_terminal_workflow" in record.get("macro_nrmse_by_method", {})
        ]
        scenario_rows.append(
            {
                "scenario": scenario,
                "replicates": len(subset),
                "median_whole_precision": float(
                    np.median([float(value["precision"]) for value in whole])
                ),
                "median_whole_recall": float(
                    np.median([float(value["recall"]) for value in whole])
                ),
                "exact_support_recovery_rate": float(
                    np.mean([bool(value["exact_support_recovery"]) for value in whole])
                ),
                "median_interaction_recall": float(
                    np.median([float(value["recall"]) for value in interactions])
                ),
                "median_proposed_workflow_nrmse": (
                    float(np.median(proposed)) if proposed else np.nan
                ),
            }
        )
    if not scenario_rows:
        raise ValueError("final recovery bundle contains no support-recovery metrics")
    scenario_summary = pd.DataFrame(scenario_rows)
    scenario_summary.to_csv(tables_dir / "recovery_scenario_summary.csv", index=False)
    scenario_rows_tex = []
    for row in scenario_rows:
        nrmse = (
            f"{row['median_proposed_workflow_nrmse']:.3f}"
            if np.isfinite(row["median_proposed_workflow_nrmse"])
            else "--"
        )
        scenario_rows_tex.append(
            f"{_tex_escape(str(row['scenario']).replace('_', ' ').title())} & "
            f"{int(row['replicates']):,} & {row['median_whole_precision']:.2f} & "
            f"{row['median_whole_recall']:.2f} & {row['median_interaction_recall']:.2f} & "
            f"{nrmse}"
        )
    (manuscript_dir / "recovery_scenario_rows.tex").write_text(
        "% Generated from final recovery replicates; do not hand-edit.\n"
        + " \\\\\n".join(scenario_rows_tex)
        + "\n",
        encoding="utf-8",
    )
    return {
        "fwer_scenario_count": len(fwer),
        "recovery_scenario_count": len(scenario_rows),
    }


def build_publication_artifacts(
    *,
    package_root: str | Path,
    results_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build all compact publication surfaces from a final confirmatory campaign."""
    package = Path(package_root).resolve()
    results = Path(results_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("publication output directory must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)
    tables = destination / "tables"
    figures = destination / "figure_data"
    final_model = destination / "final_model"
    metadata = destination / "metadata"
    manuscript = destination / "manuscript"
    provenance = destination / "provenance"
    for directory in (tables, figures, final_model, metadata, manuscript, provenance):
        directory.mkdir()

    summary = _read_json(package / "package_summary.json")
    if summary.get("package_mode") != "confirmatory":
        raise ValueError(
            "publication artifacts require a confirmatory campaign package"
        )
    run_id = str(summary["run_id"])
    contract_hash = str(summary["config_hash"])
    if re.fullmatch(r"[0-9a-f]{64}", contract_hash) is None:
        raise ValueError("campaign contract hash is malformed")

    consumed_stages = (
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
    for stage in consumed_stages:
        _verify_completed_reducer(
            package_summary=summary,
            results_root=results,
            stage=stage,
            contract_hash=contract_hash,
        )

    inventory_path = Path(str(summary.get("campaign_inventory_path", "")))
    if not inventory_path.is_file() or _sha256(inventory_path) != summary.get(
        "campaign_inventory_hash"
    ):
        raise ValueError("campaign inventory is absent or hash-mismatched")
    inventory_record_count = sum(
        bool(line.strip())
        for line in inventory_path.read_text(encoding="utf-8").splitlines()
    )
    contract_path = package / "contract" / "g11_campaign_contract.toml"
    from rfm_pipeline.campaign_contract import load_contract

    contract, loaded_contract_hash = load_contract(contract_path)
    if loaded_contract_hash != contract_hash:
        raise ValueError(
            "packaged campaign contract differs from its semantic identity"
        )
    if contract.fixed_family_replicates != 300:
        raise ValueError(
            "publication contract does not use the frozen 300-replicate supplement"
        )
    amendment_path = package / "contract" / "fixed_family_amendment.json"
    amendment = _read_json(amendment_path)
    amendment_identity = {
        key: value
        for key, value in amendment.items()
        if key != "contract_amendment_sha256"
    }
    amendment_hash = hashlib.sha256(
        json.dumps(
            amendment_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    generation_transition = {
        "superseded_schema_version": "g11_campaign_contract_v10",
        "superseded_generation": 11,
        "superseded_contract_hash": (
            "66f9a7fb702c0726c464393f7c153001846510d891a75ccb2c1d48b467375c24"
        ),
        "selected_schema_version": contract.schema_version,
        "selected_generation": int(contract.generation),
        "changed_scientific_fields": {
            "tree_family_alpha": {"before": 0.025, "after": 0.020},
            "binary_binary_family_alpha": {"before": 0.025, "after": 0.020},
            "fixed_family_replicates": {"before": 200, "after": 300},
        },
        "superseded_gate_b_evidence_status": "development_only_not_adoptable",
        "superseded_fixed_family_evidence_status": "failed_confirmatory_not_reusable",
        "required_fresh_phases": ["gate_b", "fixed_family_supplement"],
        "seed_policy": "contract_hash_derived_zero_overlap_required",
    }
    if (
        amendment.get("contract_amendment_sha256") != amendment_hash
        or amendment.get("changed_fields")
        != {"fixed_family_replicates": {"before": 1000, "after": 300}}
        or amendment.get("primary_null_regimes")
        != {"count": 5, "replicates_per_regime": 1000, "unchanged": True}
        or amendment.get("selected_contract_hash") != contract_hash
        or amendment.get("scientific_generation_transition")
        != generation_transition
    ):
        raise ValueError("fixed-family publication amendment is stale or malformed")
    shutil.copy2(contract_path, provenance / "g11_campaign_contract.toml")
    shutil.copy2(amendment_path, provenance / "fixed_family_amendment.json")
    public_summary = {
        key: summary[key]
        for key in (
            "run_id",
            "package_mode",
            "source_hash",
            "config_hash",
            "lock_hash",
            "campaign_inventory_hash",
            "campaign_envelope",
        )
    }
    public_summary["campaign_inventory_record_count"] = inventory_record_count
    public_summary["stages"] = [
        {
            key: row[key]
            for key in ("name", "partition", "job_count", "reducer_output_hash")
        }
        for row in summary["stages"]
    ]
    _write_json(
        provenance / "campaign_package_summary.json",
        public_summary,
    )

    applied_stage = next(
        row for row in summary["stages"] if row.get("name") == "applied_conditioning"
    )
    records = [
        json.loads(line)
        for line in Path(str(applied_stage["manifest_path"]))
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("applied conditioning package manifest is empty")
    provenance_fields = (
        "applied_data_root",
        "applied_data_manifest_sha256",
        "applied_config_path",
        "applied_config_sha256",
        "scientific_adapter_path",
        "scientific_adapter_sha256",
    )
    if any(
        len({str(record.get(field, "")) for record in records}) != 1
        for field in provenance_fields
    ):
        raise ValueError("applied conditioning manifest mixes provenance identities")
    applied = records[0]
    applied_data_root = Path(str(applied["applied_data_root"]))
    dataset_manifest_path = applied_data_root / "dataset_manifest.json"
    if _sha256(dataset_manifest_path) != str(applied["applied_data_manifest_sha256"]):
        raise ValueError("applied dataset manifest file checksum differs")
    dataset_manifest = _read_json(dataset_manifest_path)
    dataset_identity = {
        key: value
        for key, value in dataset_manifest.items()
        if key != "dataset_manifest_sha256"
    }
    dataset_identity_hash = hashlib.sha256(
        json.dumps(dataset_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        dataset_manifest.get("status") != "PREPARED_HOLDOUT_SEALED"
        or dataset_manifest.get("dataset_manifest_sha256") != dataset_identity_hash
        or dataset_manifest.get("dimensions")
        != {
            "rows": 30_000,
            "train_rows": 28_500,
            "holdout_rows": 1_500,
            "inputs": 160,
            "continuous_inputs": 158,
            "binary_inputs": 2,
            "outputs": 23_495,
        }
    ):
        raise ValueError("applied dataset manifest has stale or invalid dimensions")
    public_dataset_manifest = {
        key: value
        for key, value in dataset_manifest.items()
        if key not in {"source_root", "adaptive_stage_root", "holdout_stage_root"}
    }
    public_dataset_manifest["source_dataset_manifest_file_sha256"] = str(
        applied["applied_data_manifest_sha256"]
    )
    _write_json(provenance / "applied_dataset_manifest.json", public_dataset_manifest)
    catalog_relative = "metadata/manuscript_feature_catalog.parquet"
    catalog_source = applied_data_root / catalog_relative
    catalog_hash = dataset_manifest.get("generated_sha256", {}).get(catalog_relative)
    _copy_verified(
        catalog_source,
        metadata / "manuscript_feature_catalog.parquet",
        expected_sha256=str(catalog_hash),
    )
    input_catalog = pd.read_parquet(catalog_source)
    if (
        len(input_catalog) != 160
        or not {"feature_name", "feature_type"} <= set(input_catalog.columns)
        or input_catalog["feature_name"].astype(str).duplicated().any()
        or set(input_catalog["feature_type"].astype(str)) != {"first_order"}
    ):
        raise ValueError(
            "applied input feature catalog does not cover 160 first-order inputs"
        )
    binary_names = {
        "FM.Use Agnostic FS Conversion",
        "OI.Use AEO Reference Oil",
    }
    observed_names = set(input_catalog["feature_name"].astype(str))
    if not binary_names <= observed_names:
        raise ValueError("applied input feature catalog omits a binary scenario input")
    input_catalog = input_catalog.copy()
    input_catalog["is_binary"] = (
        input_catalog["feature_name"].astype(str).isin(binary_names)
    )
    input_catalog.to_csv(metadata / "input_feature_catalog.csv", index=False)
    _copy_verified(
        Path(str(applied["applied_config_path"])),
        provenance / "manuscript_case_study.yml",
        expected_sha256=str(applied["applied_config_sha256"]),
    )
    _copy_verified(
        Path(str(applied["scientific_adapter_path"])),
        provenance / "g11_campaign_adapter.py",
        expected_sha256=str(applied["scientific_adapter_sha256"]),
    )

    conditioning = _stage_summary(results, "applied_conditioning")
    screening = _stage_summary(results, "applied_screening")
    interaction = _stage_summary(results, "applied_interaction")
    nonlinear = _stage_summary(results, "applied_nonlinear")
    sparse = _stage_summary(results, "applied_sparse_resample")
    terminal_summary = _stage_summary(results, "applied_terminal_fit")
    bootstrap = _stage_summary(results, "applied_bootstrap")
    if bootstrap.get("contract_hash") != contract_hash:
        raise ValueError("bootstrap summary belongs to another contract")
    if (
        int(conditioning.get("retained_output_count", -1)) != _EXPECTED_ELIGIBLE_OUTPUTS
        or int(conditioning.get("culled_output_count", -1))
        != _EXPECTED_EXCLUDED_OUTPUTS
        or int(bootstrap.get("eligible_count", -1)) != _EXPECTED_ELIGIBLE_OUTPUTS
    ):
        raise ValueError("applied output accounting differs from 9,954 / 13,541")

    eligibility_path = (
        _stage_reducer(results, "applied_eligibility") / "output_eligibility_ledger.csv"
    )
    eligibility = pd.read_csv(eligibility_path)
    if len(eligibility) != _EXPECTED_TOTAL_OUTPUTS or "eligible" not in eligibility:
        raise ValueError(
            "output eligibility ledger does not contain exactly 23,495 rows"
        )
    eligibility["eligible"] = _strict_bool_series(
        eligibility["eligible"], name="output eligibility ledger"
    )
    from rfm_pipeline.manuscript_stages import (
        validate_g11_production_eligibility_ledger,
    )

    validate_g11_production_eligibility_ledger(
        eligibility,
        expected_accounting={
            "total": _EXPECTED_TOTAL_OUTPUTS,
            "eligible": _EXPECTED_ELIGIBLE_OUTPUTS,
            "excluded": _EXPECTED_EXCLUDED_OUTPUTS,
        },
    )
    eligible = eligibility["eligible"].to_numpy(dtype=bool)
    shutil.copy2(eligibility_path, tables / "output_eligibility_ledger.csv")

    terminal = _stage_reducer(results, "applied_terminal_fit") / "terminal"
    terminal_manifest = _read_json(terminal / "terminal_manifest.json")
    freeze_manifest_path = terminal / "model" / "freeze_manifest.json"
    freeze_manifest = _read_json(freeze_manifest_path)
    model_path = terminal / "model" / "frozen_model.npz"
    table_hashes = terminal_manifest.get("table_sha256")
    if (
        terminal_manifest.get("model_manifest_sha256") != _sha256(freeze_manifest_path)
        or terminal_manifest.get("model_npz_sha256") != _sha256(model_path)
        or not isinstance(table_hashes, dict)
    ):
        raise ValueError("terminal manifest does not bind the frozen model bundle")
    for name, expected_hash in table_hashes.items():
        if _sha256(terminal / str(name)) != expected_hash:
            raise ValueError(f"terminal table checksum differs: {name}")

    from rfm_pipeline.manuscript_stages import _load_train_fit_and_freeze

    frozen_model = _load_train_fit_and_freeze(terminal / "model")
    feature_names = [str(value) for value in terminal_manifest["final_feature_names"]]
    output_names = list(frozen_model.freeze_manifest.output_names)
    if len(output_names) != _EXPECTED_TOTAL_OUTPUTS or len(set(output_names)) != len(
        output_names
    ):
        raise ValueError(
            "frozen model output identity does not cover 23,495 unique outputs"
        )
    if (
        freeze_manifest.get("contract_hash") != contract_hash
        or terminal_manifest.get("freeze_hash")
        != frozen_model.freeze_manifest.freeze_hash
        or terminal_summary.get("freeze_hash")
        != frozen_model.freeze_manifest.freeze_hash
        or tuple(feature_names) != frozen_model.freeze_manifest.feature_names
    ):
        raise ValueError(
            "terminal model identity differs from the final campaign contract"
        )
    if int(terminal_summary.get("final_count", -1)) != len(feature_names):
        raise ValueError("terminal reducer count differs from its frozen model")
    if (
        "output_id" in eligibility.columns
        and eligibility["output_id"].astype(str).tolist() != output_names
    ):
        raise ValueError("eligibility output order differs from the frozen model")

    coefficients = frozen_model.coef
    intercept = frozen_model.intercept
    x_means = frozen_model.x_means
    x_scales = frozen_model.x_scales
    if np.any(x_scales <= 0) or not np.isfinite(coefficients).all():
        raise ValueError("frozen coefficient bundle contains invalid numeric values")
    raw_coefficients = coefficients / x_scales[:, None]
    raw_intercept = intercept - (x_means / x_scales) @ coefficients
    _write_wide_matrix(
        final_model / "coefficient_matrix_standardized.csv",
        feature_names=feature_names,
        output_names=output_names,
        values=coefficients,
    )
    _write_wide_matrix(
        final_model / "coefficient_matrix_raw_scale.csv",
        feature_names=feature_names,
        output_names=output_names,
        values=raw_coefficients,
    )
    pd.DataFrame(
        {"feature_name": feature_names, "mean": x_means, "scale": x_scales}
    ).to_csv(final_model / "x_standardization.csv", index=False)
    pd.DataFrame({"output_name": output_names, "intercept": raw_intercept}).to_csv(
        final_model / "per_output_intercepts.csv", index=False
    )
    support = pd.DataFrame(
        {
            "final_support_position": range(len(feature_names)),
            "feature_name": feature_names,
            "feature_type": [_feature_type(name) for name in feature_names],
        }
    )
    support.to_csv(final_model / "final_support_features.csv", index=False)
    pd.DataFrame(
        {
            "prefilter_support_position": range(
                len(terminal_manifest["prefilter_feature_names"])
            ),
            "feature_name": terminal_manifest["prefilter_feature_names"],
        }
    ).to_csv(final_model / "prefilter_support_features.csv", index=False)
    for name in (
        "hc3_inferential_filter_summary.csv",
        "feature_pruning_impact.csv",
        "feature_pruning_summary.csv",
    ):
        shutil.copy2(terminal / name, final_model / name)
    shutil.copy2(
        terminal / "feature_pruning_summary.csv", tables / "feature_pruning_summary.csv"
    )
    shutil.copy2(model_path, final_model / "frozen_model.npz")
    shutil.copy2(freeze_manifest_path, final_model / "freeze_manifest.json")
    shutil.copy2(
        terminal / "terminal_manifest.json", final_model / "terminal_manifest.json"
    )

    predictions_root = (
        _stage_reducer(results, "applied_holdout_predict") / "models" / "final_ols"
    )
    from rfm_pipeline.manuscript_stages import read_frozen_prediction_matrices

    frozen_predictions = read_frozen_prediction_matrices(predictions_root)
    if (
        frozen_predictions.freeze_hash != frozen_model.freeze_manifest.freeze_hash
        or list(frozen_predictions.output_names) != output_names
    ):
        raise ValueError("frozen holdout predictions differ from the terminal model")
    comparison_columns = (
        "output_id",
        "ref_min",
        "ref_max",
        "ref_mean",
        "ref_variance",
        "eligible",
    )
    expected_eligibility = frozen_predictions.eligibility_ledger.loc[
        :, comparison_columns
    ].reset_index(drop=True)
    observed_eligibility = eligibility.loc[:, comparison_columns].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            observed_eligibility,
            expected_eligibility,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError(
            "eligibility ledger differs from frozen holdout evidence"
        ) from error
    truth = frozen_predictions.y_holdout
    prediction = frozen_predictions.y_pred
    ref_range = frozen_predictions.y_train_max - frozen_predictions.y_train_min
    if truth.shape != prediction.shape or truth.shape[1] != _EXPECTED_TOTAL_OUTPUTS:
        raise ValueError("frozen final predictions do not cover all outputs")
    rmse = np.sqrt(np.mean((truth - prediction) ** 2, axis=0))
    nrmse = np.full(_EXPECTED_TOTAL_OUTPUTS, np.nan)
    nrmse[eligible] = rmse[eligible] / ref_range[eligible]
    if not np.isfinite(nrmse[eligible]).all():
        raise ValueError("eligible per-output nRMSE contains non-finite values")
    per_output = pd.DataFrame(
        {
            "output_name": output_names,
            "rmse": rmse,
            "ref_range": ref_range,
            "nrmse": nrmse,
            "included_in_macro": eligible,
        }
    )
    per_output.to_csv(tables / "per_output_nrmse.csv", index=False)
    from rfm_pipeline.manuscript_stages import (
        _build_legacy_feature_type_counts,
        _build_legacy_influential_counts_by_module,
        _build_legacy_interaction_counts_by_module_pair,
        _build_legacy_module_total_interactions,
        _build_per_output_nrmse_summary,
        _build_selected_by_module_figure_data,
        _build_support_composition_figure_data,
    )

    per_output_summary = _build_per_output_nrmse_summary(per_output)
    per_output_summary.to_csv(tables / "per_output_nrmse_summary.csv", index=False)

    feature_counts = _load_model_feature_counts(results)
    model_summaries = bootstrap.get("models")
    if not isinstance(model_summaries, dict) or set(model_summaries) != set(_MODELS):
        raise ValueError("bootstrap summary does not cover all five ablation models")
    if (
        model_summaries["final_ols"].get("freeze_hash")
        != frozen_predictions.freeze_hash
    ):
        raise ValueError(
            "final bootstrap summary differs from frozen holdout predictions"
        )
    ablation_rows = []
    for model in _MODELS:
        row = model_summaries[model]
        if int(row.get("eligible_count", -1)) != _EXPECTED_ELIGIBLE_OUTPUTS:
            raise ValueError(
                f"ablation metric for {model} uses the wrong output population"
            )
        ablation_rows.append(
            {
                "model_name": model,
                "n_features": feature_counts[model],
                "nrmse": float(row["point_estimate"]),
                "ci_lower": float(row["ci_lower"]),
                "ci_upper": float(row["ci_upper"]),
            }
        )
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(tables / "ablation_table.csv", index=False)
    ablation.to_csv(tables / "model_performance.csv", index=False)
    conditioning_summary_path = (
        _stage_reducer(results, "applied_conditioning")
        / "output_conditioning"
        / "output_conditioning_summary.csv"
    )
    conditioning_table = pd.read_csv(conditioning_summary_path)
    if (
        len(conditioning_table) != 1
        or "n_components_retained" not in conditioning_table
    ):
        raise ValueError("output conditioning does not identify its retained PCA basis")
    pca_component_count = int(conditioning_table.loc[0, "n_components_retained"])
    if pca_component_count < 1:
        raise ValueError("output conditioning retained no PCA components")

    workflow = pd.DataFrame(
        [
            (
                "output_conditioning",
                "retained_scalar_outputs",
                _EXPECTED_ELIGIBLE_OUTPUTS,
            ),
            ("output_conditioning", "retained_pca_components", pca_component_count),
            (
                "empirical_null_screening",
                "retained_terms",
                int(screening["retained_term_count"]),
            ),
            (
                "interaction_discovery",
                "candidate_pairs",
                int(interaction["candidate_pair_count"]),
            ),
            (
                "interaction_discovery",
                "retained_pairs",
                int(interaction["retained_pair_count"]),
            ),
            (
                "nonlinear_discovery",
                "retained_transformations",
                int(nonlinear["retained_transformation_count"]),
            ),
            (
                "sparse_stability",
                "retained_terms",
                int(sparse["final_stable_support_count"]),
            ),
            (
                "terminal_fit",
                "prefilter_terms",
                len(terminal_manifest["prefilter_feature_names"]),
            ),
            ("terminal_fit", "hc3_terms", len(terminal_manifest["hc3_feature_names"])),
            ("terminal_fit", "final_terms", len(feature_names)),
        ],
        columns=["stage", "primary_quantity", "recomputed_value"],
    )
    workflow.to_csv(tables / "workflow_stage_summary.csv", index=False)

    composition = _build_support_composition_figure_data(support)
    modules = _build_selected_by_module_figure_data(support)
    type_counts = _build_legacy_feature_type_counts(support)
    influential = _build_legacy_influential_counts_by_module(support)
    pairs, matrix = _build_legacy_interaction_counts_by_module_pair(support)
    totals = _build_legacy_module_total_interactions(matrix)
    composition.to_csv(figures / "figure_support_composition_data.csv", index=False)
    modules.to_csv(figures / "figure_selected_by_module_data.csv", index=False)
    type_counts.to_csv(figures / "feature_type_counts.csv", index=False)
    influential.to_csv(figures / "influential_counts_by_module.csv", index=False)
    pairs.to_csv(figures / "interaction_counts_by_module_pair.csv", index=False)
    matrix.rename_axis("module").reset_index().to_csv(
        figures / "interaction_density_module_matrix.csv", index=False
    )
    totals.to_csv(figures / "module_total_interactions.csv", index=False)
    pruning_impact = pd.read_csv(terminal / "feature_pruning_impact.csv")
    pruning_impact.to_csv(
        figures / "figure_feature_pruning_curve_data.csv", index=False
    )

    recovery_counts = _build_recovery_tables(results, tables, manuscript, contract_hash)

    main_count = int((support["feature_type"] == "First Order").sum())
    interaction_count = int((support["feature_type"] == "Second Order").sum())
    transform_count = int((support["feature_type"] == "Non-Linear").sum())
    final_metric = model_summaries["final_ols"]
    macro_values = {
        "FinalRunID": run_id.replace("_", r"\_"),
        "FinalPredictorCount": f"{len(feature_names):,}",
        "FinalMainEffectCount": f"{main_count:,}",
        "FinalInteractionCount": f"{interaction_count:,}",
        "FinalTransformationCount": f"{transform_count:,}",
        "ScreenedTermCount": f"{int(screening['retained_term_count']):,}",
        "DiscoveredInteractionCount": f"{int(interaction['retained_pair_count']):,}",
        "DiscoveredTransformationCount": f"{int(nonlinear['retained_transformation_count']):,}",
        "PCAComponentCount": f"{pca_component_count:,}",
        "SparseStablePredictorCount": f"{int(sparse['final_stable_support_count']):,}",
        "PrefilterPredictorCount": f"{len(terminal_manifest['prefilter_feature_names']):,}",
        "HCThreePredictorCount": f"{len(terminal_manifest['hc3_feature_names']):,}",
        "PrunedPredictorCount": f"{len(terminal_manifest['hc3_feature_names']) - len(feature_names):,}",
        "EligibleOutputCount": f"{_EXPECTED_ELIGIBLE_OUTPUTS:,}",
        "ExcludedOutputCount": f"{_EXPECTED_EXCLUDED_OUTPUTS:,}",
        "TotalOutputCount": f"{_EXPECTED_TOTAL_OUTPUTS:,}",
        "FinalNrmse": f"{float(final_metric['point_estimate']):.4f}",
        "FinalNrmseCILower": f"{float(final_metric['ci_lower']):.4f}",
        "FinalNrmseCIUpper": f"{float(final_metric['ci_upper']):.4f}",
        "BootstrapDrawCount": f"{int(final_metric['draw_count']):,}",
        "PerOutputNrmseMedian": f"{float(per_output_summary.loc[0, 'p50']):.3f}",
        "PerOutputNrmsePTen": f"{float(per_output_summary.loc[0, 'p10']):.3f}",
        "PerOutputNrmsePTwentyFive": f"{float(per_output_summary.loc[0, 'p25']):.3f}",
        "PerOutputNrmsePSeventyFive": f"{float(per_output_summary.loc[0, 'p75']):.3f}",
        "PerOutputNrmsePNinety": f"{float(per_output_summary.loc[0, 'p90']):.3f}",
        "PerOutputNrmseMaximum": f"{float(np.nanmax(nrmse[eligible])):.3f}",
        "CalibrationScenarioCount": f"{recovery_counts['fwer_scenario_count']:,}",
        "RecoveryScenarioCount": f"{recovery_counts['recovery_scenario_count']:,}",
        "InteractionCalibrationConclusion": (
            "All prespecified null regimes and fixed-family supplements met the "
            "one-sided calibration criterion."
        ),
    }
    macro_prefixes = {
        "null_mean": "NullMean",
        "main_effects_ols": "MainEffects",
        "screened_ols": "Screened",
        "penalized_ols": "Penalized",
        "final_ols": "Final",
    }
    for model, prefix in macro_prefixes.items():
        model_metric = model_summaries[model]
        macro_values[f"{prefix}PredictorCount"] = f"{feature_counts[model]:,}"
        macro_values[f"{prefix}Nrmse"] = f"{float(model_metric['point_estimate']):.4f}"
        macro_values[f"{prefix}NrmseCILower"] = f"{float(model_metric['ci_lower']):.4f}"
        macro_values[f"{prefix}NrmseCIUpper"] = f"{float(model_metric['ci_upper']):.4f}"
    _write_manuscript_macros(manuscript / "manuscript_results.tex", macro_values)
    _write_json(manuscript / "manuscript_results.json", macro_values)

    (destination / "README.md").write_text(
        "# G11 publication artifact bundle\n\n"
        f"This immutable bundle was compiled from confirmatory run `{run_id}`. "
        "Every file is checksummed in `publication_artifact_manifest.json`.\n\n"
        "- `tables/`: manuscript result tables and output-level diagnostics.\n"
        "- `figure_data/`: exact CSV inputs for deterministic figure regeneration.\n"
        "- `final_model/`: named support, frozen model, raw-scale and standardized-X "
        "coefficient exports, and per-output intercepts.\n"
        "- `metadata/`: the complete 160-input feature catalog in Parquet and CSV; "
        "all 23,495 output identities and eligibility reasons are in the tables ledger.\n"
        "- `manuscript/`: generated LaTeX values used by the article.\n"
        "- `provenance/`: final contract, package inventory, applied-data manifest, "
        "case-study config, and content-addressed scientific adapter.\n",
        encoding="utf-8",
    )
    (final_model / "README.md").write_text(
        "# Final reduced-form model\n\n"
        "`coefficient_matrix_standardized.csv` uses train-standardized predictors. "
        "For a raw predictor vector x, predictions can be computed either as "
        "`((x - mean) / scale) @ beta_standardized + intercept_standardized` using "
        "`x_standardization.csv` and the intercept in `frozen_model.npz`, or as "
        "`x @ beta_raw + intercept_raw` using `coefficient_matrix_raw_scale.csv` and "
        "`per_output_intercepts.csv`. Output columns follow the exact order in the "
        "freeze manifest and eligibility ledger.\n",
        encoding="utf-8",
    )
    (tables / "README.md").write_text(
        "# Publication tables\n\n"
        "The macro nRMSE population is exactly the 9,954 outputs marked eligible in "
        "`output_eligibility_ledger.csv`; the remaining 13,541 outputs are excluded by "
        "the prespecified zero-variance/relative-range rule. `recovery_replicates.csv` "
        "stores nested metric dictionaries as canonical JSON strings.\n",
        encoding="utf-8",
    )

    _validate_public_provenance(provenance)
    inventory = _artifact_inventory(destination)
    manifest_payload = {
        "schema_version": 1,
        "status": "PUBLICATION_ARTIFACTS_COMPLETE",
        "run_id": run_id,
        "source_hash": summary["source_hash"],
        "contract_hash": contract_hash,
        "lock_hash": summary["lock_hash"],
        "campaign_inventory_hash": summary["campaign_inventory_hash"],
        "output_accounting": {
            "total": _EXPECTED_TOTAL_OUTPUTS,
            "eligible": _EXPECTED_ELIGIBLE_OUTPUTS,
            "excluded": _EXPECTED_EXCLUDED_OUTPUTS,
        },
        "artifacts": inventory,
    }
    manifest_payload["artifact_set_sha256"] = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_path = destination / "publication_artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    manifest = build_publication_artifacts(
        package_root=args.package_root,
        results_root=args.results_root,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
