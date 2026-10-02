"""End-to-end orchestration for reduced-form model fitting and export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pandas as pd

from .final_ols import (
    FinalOLSFitResult,
    build_postfit_artifacts,
    canonical_postfit_artifact_names,
    fit_final_ols,
    make_holdout_nrmse_summary,
)
from .regularized_screening import ScreeningSelectionResult, fit_multitask_elastic_net_screen


@dataclass(frozen=True)
class CanonicalWorkflowRun:
    """Outputs from the canonical end-to-end reduced-form workflow.

    Parameters
    ----------
    screening_result
        Sparse-screening fit result used to choose retained features.
    final_ols_result
        Final OLS fit on the retained feature set.
    holdout_summary
        One-row holdout evaluation summary.
    artifacts
        Canonical post-fit artifact bundle including the manifest payload.
    artifact_format
        Tabular artifact format requested for export metadata.
    """

    screening_result: ScreeningSelectionResult
    final_ols_result: FinalOLSFitResult
    holdout_summary: pd.DataFrame
    artifacts: dict[str, pd.DataFrame | dict[str, Any]]
    artifact_format: str


def _resolve_artifact_format(requested: str) -> str:
    """Resolve the tabular artifact format.

    Parameters
    ----------
    requested
        Requested format: ``"auto"``, ``"parquet"``, or ``"csv"``.

    Returns
    -------
    str
        Concrete artifact format.

    Raises
    ------
    ValueError
        Raised when an unsupported format is requested.
    """
    if requested == "auto":
        return "parquet" if _supports_parquet() else "csv"
    if requested not in {"parquet", "csv"}:
        raise ValueError("artifact_format must be one of {'auto', 'parquet', 'csv'}.")
    return requested


def _supports_parquet() -> bool:
    """Return whether a parquet engine is importable in the current environment."""
    return find_spec("pyarrow") is not None or find_spec("fastparquet") is not None


def _validate_workflow_inputs(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    Y_holdout: pd.DataFrame,
) -> None:
    """Fail before fitting when labeled table alignment is ambiguous."""
    frames = {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_holdout": X_holdout,
        "Y_holdout": Y_holdout,
    }
    for name, frame in frames.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        if frame.index.has_duplicates:
            raise ValueError(f"{name} must not contain duplicate row indexes")
        if frame.columns.has_duplicates:
            raise ValueError(f"{name} must not contain duplicate column names")

    if not X_train.index.equals(Y_train.index):
        raise ValueError("X_train and Y_train must have identical row indexes in the same order")
    if not X_holdout.index.equals(Y_holdout.index):
        raise ValueError(
            "X_holdout and Y_holdout must have identical row indexes in the same order"
        )
    if not X_train.columns.equals(X_holdout.columns):
        raise ValueError("training and holdout feature columns must match exactly and in order")
    if not Y_train.columns.equals(Y_holdout.columns):
        raise ValueError("training and holdout output columns must match exactly and in order")


def run_canonical_workflow(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    Y_holdout: pd.DataFrame,
    *,
    dataset_tag: str,
    all_input_features: list[str] | None = None,
    screening_cv: int = 5,
    screening_l1_ratio: float | tuple[float, ...] = (0.9, 1.0),
    screening_alphas: int | list[float] = 100,
    screening_max_iter: int = 10_000,
    screening_selection_tol: float = 1e-6,
    screening_random_state: int = 123,
    n_boot: int = 1000,
    alpha: float = 0.05,
    bootstrap_random_state: int = 123,
    bootstrap_sample_size: int | None = None,
    artifact_format: str = "auto",
    upstream_provenance: dict[str, Any] | None = None,
) -> CanonicalWorkflowRun:
    """Run the implemented screening, final OLS, evaluation, and artifact assembly flow.

    Parameters
    ----------
    X_train
        Raw-scale training feature matrix.
    Y_train
        Raw-scale training response matrix.
    X_holdout
        Raw-scale holdout feature matrix.
    Y_holdout
        Raw-scale holdout response matrix.
    dataset_tag
        Human-readable run label stored in the manifest.
    all_input_features
        Original-order candidate input list before screening. Defaults to and,
        when supplied, must exactly match ``X_train`` column order.
    screening_cv
        Cross-validation folds for multitask elastic-net screening.
    screening_l1_ratio
        Elastic-net mixing weight(s) for screening.
    screening_alphas
        Number of alphas or explicit alpha grid for screening.
    screening_max_iter
        Maximum optimizer iterations for screening.
    screening_selection_tol
        Threshold for declaring a feature retained from the screening fit.
    screening_random_state
        Screening random seed.
    n_boot
        Holdout bootstrap replicate count.
    alpha
        Two-sided bootstrap error level.
    bootstrap_random_state
        Holdout bootstrap random seed.
    bootstrap_sample_size
        Optional bootstrap draw size.
    artifact_format
        ``"auto"``, ``"parquet"``, or ``"csv"``.
    upstream_provenance
        Optional metadata merged into the manifest provenance payload.

    Returns
    -------
    CanonicalWorkflowRun
        End-to-end screening, fitting, evaluation, and artifact bundle outputs.

    Raises
    ------
    TypeError
        Raised when an input is not a pandas DataFrame.
    ValueError
        Raised when labels are duplicated or misaligned, the dataset tag is
        empty, or screening retains no features.
    """
    _validate_workflow_inputs(X_train, Y_train, X_holdout, Y_holdout)
    if not isinstance(dataset_tag, str) or not dataset_tag.strip():
        raise ValueError("dataset_tag must be a non-empty string")
    feature_names = [str(column) for column in X_train.columns]
    resolved_all_features = list(all_input_features or feature_names)
    if resolved_all_features != feature_names:
        raise ValueError("all_input_features must match X_train columns exactly and in order")
    resolved_format = _resolve_artifact_format(artifact_format)

    screening_result = fit_multitask_elastic_net_screen(
        X_train,
        Y_train,
        cv=screening_cv,
        l1_ratio=screening_l1_ratio,
        alphas=screening_alphas,
        max_iter=screening_max_iter,
        selection_tol=screening_selection_tol,
        random_state=screening_random_state,
    )
    selected_features = list(screening_result.selected_features)
    if not selected_features:
        raise ValueError("Screening retained no features; final OLS cannot be fit.")

    final_result = fit_final_ols(X_train.loc[:, selected_features], Y_train)
    holdout_summary = make_holdout_nrmse_summary(
        final_result,
        X_holdout.loc[:, selected_features],
        Y_holdout,
        Y_ref=Y_train,
        n_boot=n_boot,
        alpha=alpha,
        random_state=bootstrap_random_state,
        sample_size=bootstrap_sample_size,
    )

    summary_record = holdout_summary.iloc[0].to_dict()
    metrics = {"holdout_nrmse": summary_record}
    evaluation = {
        "artifact_format": resolved_format,
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "bootstrap_sample_size": int(
            bootstrap_sample_size if bootstrap_sample_size is not None else len(Y_holdout)
        ),
        "holdout_rows": int(len(Y_holdout)),
        "screening_cv": int(screening_result.cv_folds),
        "screening_alpha": float(screening_result.alpha_),
        "n_selected_features": int(len(selected_features)),
    }
    provenance = {
        "workflow_stage": "evaluation_export",
        "screening_estimator": "MultiTaskElasticNetCV",
        "final_estimator": "LinearRegression",
        **dict(upstream_provenance or {}),
    }
    artifacts = build_postfit_artifacts(
        final_result,
        dataset_tag=dataset_tag,
        all_input_features=resolved_all_features,
        selected_features=selected_features,
        evaluation_summary=holdout_summary,
        upstream_provenance=provenance,
        metrics=metrics,
        evaluation=evaluation,
        artifact_format=resolved_format,
    )

    return CanonicalWorkflowRun(
        screening_result=screening_result,
        final_ols_result=final_result,
        holdout_summary=holdout_summary,
        artifacts=artifacts,
        artifact_format=resolved_format,
    )


def write_postfit_bundle(
    artifacts: dict[str, pd.DataFrame | dict[str, Any]],
    root: Path,
) -> dict[str, Path]:
    """Write a canonical post-fit bundle to disk.

    Parameters
    ----------
    artifacts
        Artifact bundle produced by :func:`run_canonical_workflow` or
        :func:`rfm_pipeline.final_ols.build_postfit_artifacts`.
    root
        Destination bundle directory.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping from logical artifact names to the written on-disk paths.
    """
    bundle_root = Path(root).resolve()
    bundle_root.mkdir(parents=True, exist_ok=True)

    manifest = dict(artifacts.get("manifest", {}))
    file_map = dict(manifest.get("files", {}))
    written: dict[str, Path] = {}

    for artifact_name in canonical_postfit_artifact_names():
        artifact = artifacts[artifact_name]
        if not isinstance(artifact, pd.DataFrame):
            raise TypeError(f"Artifact '{artifact_name}' must be a pandas DataFrame.")
        requested_rel = file_map.get(artifact_name, f"postfit_diagnostics/{artifact_name}.parquet")
        actual_path = _write_artifact_table(artifact, bundle_root, Path(requested_rel))
        written[artifact_name] = actual_path
        file_map[artifact_name] = actual_path.relative_to(bundle_root).as_posix()

    manifest["files"] = file_map
    manifest_path = bundle_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    written["manifest"] = manifest_path
    return written


def _write_artifact_table(frame: pd.DataFrame, root: Path, relative_path: Path) -> Path:
    """Write one artifact table using parquet when available, otherwise CSV."""
    if relative_path.is_absolute():
        raise ValueError("Artifact path resolves outside the bundle directory.")
    destination = (root / relative_path).resolve()
    if root != destination and root not in destination.parents:
        raise ValueError("Artifact path resolves outside the bundle directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".parquet":
        try:
            frame.to_parquet(destination, index=False)
            return destination
        except (ImportError, ModuleNotFoundError, ValueError):
            destination = destination.with_suffix(".csv")
    elif destination.suffix != ".csv":
        raise ValueError("Artifact tables must use a .parquet or .csv extension.")

    frame.to_csv(destination, index=False)
    return destination
