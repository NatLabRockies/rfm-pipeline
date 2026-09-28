"""Independently audit a compiled G11 publication bundle and manuscript PDFs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:  # Support direct CLI execution from the repository.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.reproduce_artifacts import (
    MANUSCRIPT_FIGURES,
    validate_manuscript_figure_pdfs,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _strict_bool(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    values = series.astype("string")
    if values.isna().any() or not values.isin(("True", "False")).all():
        raise ValueError(f"{name} must contain only explicit True/False values")
    return values.map({"True": True, "False": False}).astype(bool)


def _verify_wide_matrix(
    path: Path,
    *,
    expected_features: list[str],
    expected_outputs: list[str],
    expected_values: np.ndarray,
    value_label: str,
) -> None:
    if expected_values.shape != (len(expected_features), len(expected_outputs)):
        raise ValueError(f"{value_label} expected matrix shape differs")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["feature_name", *expected_outputs]:
            raise ValueError(f"coefficient output identity differs: {path.name}")
        observed_features = []
        for index, row in enumerate(reader):
            if len(row) != len(header):
                raise ValueError(f"coefficient row width differs: {path.name}")
            if index >= len(expected_features):
                raise ValueError(f"coefficient row count differs: {path.name}")
            observed_features.append(row[0])
            try:
                observed = np.asarray(row[1:], dtype=np.float64)
            except ValueError as exc:
                raise ValueError(f"{value_label} contains nonnumeric values") from exc
            if not np.allclose(
                observed,
                expected_values[index],
                rtol=1.0e-12,
                atol=1.0e-12,
                equal_nan=False,
            ):
                raise ValueError(f"{value_label} values differ from the frozen model")
    if observed_features != expected_features:
        raise ValueError(f"coefficient feature identity differs: {path.name}")


def audit_publication_artifacts(
    *,
    publication_root: str | Path,
    figures_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a deterministic PASS record or fail on any mixed/stale surface."""
    publication = Path(publication_root).resolve()
    figures = Path(figures_root).resolve()
    manifest_path = publication / "publication_artifact_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "PUBLICATION_ARTIFACTS_COMPLETE":
        raise ValueError("publication artifact compiler did not report completion")

    inventory = manifest.get("artifacts")
    if not isinstance(inventory, list):
        raise ValueError("publication manifest has no artifact inventory")
    expected = {str(row["relative_path"]): row for row in inventory}
    if len(expected) != len(inventory):
        raise ValueError("publication manifest duplicates artifact paths")
    actual = {
        path.relative_to(publication).as_posix(): path
        for path in publication.rglob("*")
        if path.is_file() and path.name != manifest_path.name
    }
    if set(actual) != set(expected):
        raise ValueError("publication directory differs from its artifact inventory")
    for relative, path in actual.items():
        row = expected[relative]
        if _sha256(path) != row.get("sha256") or path.stat().st_size != int(
            row.get("bytes", -1)
        ):
            raise ValueError(f"publication artifact checksum differs: {relative}")
    if manifest.get("artifact_set_sha256") != _stable_hash(inventory):
        raise ValueError("publication artifact-set hash differs")

    accounting = manifest.get("output_accounting")
    if accounting != {"total": 23495, "eligible": 9954, "excluded": 13541}:
        raise ValueError("publication manifest output accounting differs")
    from rfm_pipeline.campaign_contract import load_contract

    contract, contract_hash = load_contract(
        publication / "provenance" / "g11_campaign_contract.toml"
    )
    amendment = _read_json(publication / "provenance" / "fixed_family_amendment.json")
    amendment_identity = {
        key: value
        for key, value in amendment.items()
        if key != "contract_amendment_sha256"
    }
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
        contract_hash != manifest.get("contract_hash")
        or contract.fixed_family_replicates != 300
        or len([scenario for scenario in contract.scenarios if scenario.kind == "null"])
        != 5
        or any(
            scenario.n_replicates != 1000
            for scenario in contract.scenarios
            if scenario.kind == "null"
        )
        or amendment.get("contract_amendment_sha256")
        != _stable_hash(amendment_identity)
        or amendment.get("changed_fields")
        != {"fixed_family_replicates": {"before": 1000, "after": 300}}
        or amendment.get("primary_null_regimes")
        != {"count": 5, "replicates_per_regime": 1000, "unchanged": True}
        or amendment.get("selected_contract_hash") != contract_hash
        or amendment.get("scientific_generation_transition")
        != generation_transition
    ):
        raise ValueError("fixed-family publication amendment is stale or malformed")
    eligibility = pd.read_csv(publication / "tables" / "output_eligibility_ledger.csv")
    eligibility["eligible"] = _strict_bool(eligibility["eligible"], name="eligibility")
    if len(eligibility) != 23495 or int(eligibility["eligible"].sum()) != 9954:
        raise ValueError("publication eligibility ledger accounting differs")
    output_names = eligibility["output_id"].astype(str).tolist()
    if len(set(output_names)) != 23495:
        raise ValueError("publication eligibility ledger duplicates output IDs")

    input_catalog = pd.read_csv(publication / "metadata" / "input_feature_catalog.csv")
    parquet_catalog = pd.read_parquet(
        publication / "metadata" / "manuscript_feature_catalog.parquet"
    )
    try:
        pd.testing.assert_frame_equal(
            input_catalog.loc[:, list(parquet_catalog.columns)],
            parquet_catalog,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError(
            "publication input catalog CSV differs from Parquet"
        ) from error
    input_catalog["is_binary"] = _strict_bool(
        input_catalog["is_binary"], name="input catalog binary indicator"
    )
    binary_names = {
        "FM.Use Agnostic FS Conversion",
        "OI.Use AEO Reference Oil",
    }
    if (
        len(input_catalog) != 160
        or input_catalog["feature_name"].astype(str).duplicated().any()
        or set(input_catalog["feature_type"].astype(str)) != {"first_order"}
        or set(
            input_catalog.loc[input_catalog["is_binary"], "feature_name"].astype(str)
        )
        != binary_names
        or int((~input_catalog["is_binary"]).sum()) != 158
    ):
        raise ValueError("publication input catalog does not encode 158 + 2 inputs")

    support = pd.read_csv(publication / "final_model" / "final_support_features.csv")
    feature_names = support["feature_name"].astype(str).tolist()
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("publication final support duplicates feature IDs")
    freeze_manifest = _read_json(publication / "final_model" / "freeze_manifest.json")
    model_path = publication / "final_model" / "frozen_model.npz"
    if _sha256(model_path) != freeze_manifest.get("model_npz_sha256"):
        raise ValueError("publication frozen-model checksum differs")
    if (
        list(map(str, freeze_manifest.get("feature_names", []))) != feature_names
        or list(map(str, freeze_manifest.get("output_names", []))) != output_names
    ):
        raise ValueError("publication frozen-model identities differ")
    with np.load(model_path, allow_pickle=False) as arrays:
        standardized_values = np.asarray(arrays["coef"], dtype=np.float64)
        standardized_intercepts = np.asarray(arrays["intercept"], dtype=np.float64)
        x_means = np.asarray(arrays["x_means"], dtype=np.float64)
        x_scales = np.asarray(arrays["x_scales"], dtype=np.float64)
    if (
        standardized_values.shape != (len(feature_names), len(output_names))
        or standardized_intercepts.shape != (len(output_names),)
        or x_means.shape != (len(feature_names),)
        or x_scales.shape != (len(feature_names),)
        or not np.isfinite(standardized_values).all()
        or not np.isfinite(standardized_intercepts).all()
        or not np.isfinite(x_means).all()
        or not np.isfinite(x_scales).all()
        or np.any(x_scales <= 0)
    ):
        raise ValueError("publication frozen-model arrays are invalid")
    raw_values = standardized_values / x_scales[:, None]
    raw_intercepts = (
        standardized_intercepts - (x_means / x_scales) @ standardized_values
    )
    _verify_wide_matrix(
        publication / "final_model" / "coefficient_matrix_standardized.csv",
        expected_features=feature_names,
        expected_outputs=output_names,
        expected_values=standardized_values,
        value_label="standardized coefficient",
    )
    _verify_wide_matrix(
        publication / "final_model" / "coefficient_matrix_raw_scale.csv",
        expected_features=feature_names,
        expected_outputs=output_names,
        expected_values=raw_values,
        value_label="raw-scale coefficient",
    )
    standardization = pd.read_csv(publication / "final_model" / "x_standardization.csv")
    if (
        standardization["feature_name"].astype(str).tolist() != feature_names
        or not np.allclose(
            standardization["mean"].to_numpy(dtype=float),
            x_means,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        or not np.allclose(
            standardization["scale"].to_numpy(dtype=float),
            x_scales,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    ):
        raise ValueError("publication standardization values differ from the model")
    intercepts = pd.read_csv(publication / "final_model" / "per_output_intercepts.csv")
    if intercepts["output_name"].astype(
        str
    ).tolist() != output_names or not np.allclose(
        intercepts["intercept"].to_numpy(dtype=float),
        raw_intercepts,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("publication intercept output identity differs")

    per_output = pd.read_csv(publication / "tables" / "per_output_nrmse.csv")
    per_output["included_in_macro"] = _strict_bool(
        per_output["included_in_macro"], name="per-output macro membership"
    )
    if per_output["output_name"].astype(str).tolist() != output_names or not per_output[
        "included_in_macro"
    ].equals(eligibility["eligible"]):
        raise ValueError("per-output metric population differs from eligibility")

    ablation = pd.read_csv(publication / "tables" / "ablation_table.csv")
    if ablation["model_name"].astype(str).tolist() != [
        "null_mean",
        "main_effects_ols",
        "screened_ols",
        "penalized_ols",
        "final_ols",
    ]:
        raise ValueError("ablation table does not contain the frozen five-model family")

    macro_json = _read_json(publication / "manuscript" / "manuscript_results.json")
    macro_tex = (publication / "manuscript" / "manuscript_results.tex").read_text(
        encoding="utf-8"
    )
    required_macros = {
        "FinalPredictorCount",
        "EligibleOutputCount",
        "ExcludedOutputCount",
        "TotalOutputCount",
        "FinalNrmse",
        "FinalNrmseCILower",
        "FinalNrmseCIUpper",
        "BootstrapDrawCount",
        "PCAComponentCount",
        "InteractionCalibrationConclusion",
    }
    if not required_macros <= set(macro_json):
        raise ValueError("manuscript macro bundle is incomplete")
    for name, value in macro_json.items():
        if rf"\newcommand{{\{name}}}{{{value}}}" not in macro_tex:
            raise ValueError(f"LaTeX macro differs from its JSON value: {name}")
    for name in ("recovery_fwer_rows.tex", "recovery_scenario_rows.tex"):
        path = publication / "manuscript" / name
        if not path.is_file() or path.stat().st_size < 80:
            raise ValueError(f"generated manuscript recovery rows are absent: {name}")

    validate_manuscript_figure_pdfs(figures)
    figure_hashes = {name: _sha256(figures / name) for name in MANUSCRIPT_FIGURES}
    identity = {
        "schema_version": 1,
        "status": "PUBLICATION_ARTIFACT_AUDIT_PASS",
        "run_id": manifest["run_id"],
        "source_hash": manifest["source_hash"],
        "contract_hash": manifest["contract_hash"],
        "lock_hash": manifest["lock_hash"],
        "campaign_inventory_hash": manifest["campaign_inventory_hash"],
        "contract_amendment_sha256": amendment["contract_amendment_sha256"],
        "publication_manifest_sha256": _sha256(manifest_path),
        "artifact_set_sha256": manifest["artifact_set_sha256"],
        "figure_sha256": figure_hashes,
        "output_accounting": accounting,
        "final_predictor_count": len(feature_names),
    }
    result = {**identity, "audit_sha256": _stable_hash(identity)}
    if output_path is not None:
        destination = Path(output_path)
        if destination.exists():
            raise ValueError(f"refusing to overwrite publication audit: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-root", required=True)
    parser.add_argument("--figures-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = audit_publication_artifacts(
        publication_root=args.publication_root,
        figures_root=args.figures_root,
        output_path=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
