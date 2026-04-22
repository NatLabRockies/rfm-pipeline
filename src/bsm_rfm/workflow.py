"""Canonical workflow provenance and executable orchestration helpers.

This module preserves the audited workflow-stage and case-study provenance while also
providing a tested end-to-end orchestration layer that ties together the implemented
screening, final OLS, evaluation, and export foundations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
class WorkflowStage:
    """Recovered stage-level provenance for the reduced-form workflow.

    Parameters
    ----------
    name
        Stable stage identifier.
    order
        Canonical order in the recovered end-to-end workflow.
    provenance
        Provenance label describing the strongest available source for the stage.
    source_artifact
        Audited script, notebook, or report artifact supporting the stage.
    status
        Current implementation status in the live package.
    description
        Concise description of the stage responsibility.
    """

    name: str
    order: int
    provenance: str
    source_artifact: str
    status: str
    description: str


@dataclass(frozen=True)
class CaseStudyNumber:
    """Recovered quantitative fact from the audited workflow archive.

    Parameters
    ----------
    key
        Stable identifier for the recovered quantity.
    value
        Numeric value recovered from the audited source materials.
    unit
        Human-readable unit or count label.
    provenance
        Provenance label for the number.
    note
        Short explanation of how the quantity fits into the workflow.
    """

    key: str
    value: int
    unit: str
    provenance: str
    note: str


@dataclass(frozen=True)
class CanonicalWorkflowRun:
    """Outputs from the canonical end-to-end reduced-form workflow foundation.

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


def canonical_workflow_stages() -> tuple[WorkflowStage, ...]:
    """Return the recovered canonical workflow-stage sequence.

    Returns
    -------
    tuple of WorkflowStage
        Ordered stage definitions recovered from the audited archive and design notes.
    """
    return (
        WorkflowStage(
            name="upstream_null_screening",
            order=1,
            provenance="source-derived",
            source_artifact="null_distribution.py",
            status="implemented_adapter",
            description="Permutation-null Delta screening over the upstream 300k sample.",
        ),
        WorkflowStage(
            name="feature_expansion",
            order=2,
            provenance="notebook-derived",
            source_artifact="make_nonlinear_features.ipynb",
            status="spec_recovered_not_fully_ported",
            description=("Create scenario flags, nonlinear terms, and second-order interactions."),
        ),
        WorkflowStage(
            name="modeling_subset_creation",
            order=3,
            provenance="audit-resolved",
            source_artifact="workflow_audit.md",
            status="implemented_foundation",
            description=(
                "Balanced 20k modeling subset built by sampling 5k rows in each boolean "
                "scenario stratum."
            ),
        ),
        WorkflowStage(
            name="regularized_screening",
            order=4,
            provenance="notebook-derived",
            source_artifact="LASSO_to_OLS_v9.ipynb",
            status="implemented_foundation",
            description="Sparse screening and dimensionality reduction before final OLS.",
        ),
        WorkflowStage(
            name="final_ols",
            order=5,
            provenance="notebook-derived",
            source_artifact="LASSO_to_OLS_v9.ipynb",
            status="implemented_foundation",
            description="Fit interpretable output-wise OLS models on the retained feature set.",
        ),
        WorkflowStage(
            name="evaluation_export",
            order=6,
            provenance="notebook-derived",
            source_artifact="LASSO_to_OLS_v9.ipynb",
            status="implemented_foundation",
            description=(
                "Compute holdout bootstrap summaries and assemble canonical post-fit exports."
            ),
        ),
        WorkflowStage(
            name="downstream_visualization",
            order=7,
            provenance="consumer-contract",
            source_artifact="viz_io consumer expectations",
            status="implemented_foundation",
            description="Read canonical exported artifacts for postfit visualization.",
        ),
    )


def workflow_scope_boundary_table() -> pd.DataFrame:
    """Return the current implementation-boundary stages as a table.

    The canonical workflow intentionally exposes a mixed provenance boundary while the
    package remains honest about what is implemented directly versus what is still
    represented through recovered notebook- or source-derived contracts.

    Returns
    -------
    pandas.DataFrame
        Subset of :func:`workflow_stage_table` containing the stages whose implementation
        status is not yet ``"implemented_foundation"``.
    """
    table = workflow_stage_table()
    return table.loc[
        table["status"] != "implemented_foundation",
        ["order", "name", "provenance", "source_artifact", "status", "description"],
    ].reset_index(drop=True)


def workflow_stage_table() -> pd.DataFrame:
    """Return the canonical workflow-stage sequence as a table.

    Returns
    -------
    pandas.DataFrame
        One row per recovered stage ordered by the canonical workflow sequence.
    """
    return pd.DataFrame(asdict(stage) for stage in canonical_workflow_stages()).sort_values(
        "order",
        ignore_index=True,
    )


def canonical_case_study_numbers() -> tuple[CaseStudyNumber, ...]:
    """Return recovered quantitative facts for the audited case study.

    Returns
    -------
    tuple of CaseStudyNumber
        Stable, importable case-study counts that are already documented in the audit.
    """
    return (
        CaseStudyNumber(
            key="upstream_sample_size",
            value=300_000,
            unit="rows",
            provenance="source-derived",
            note="Standardized upstream sample used for null screening.",
        ),
        CaseStudyNumber(
            key="modeling_subset_size",
            value=20_000,
            unit="rows",
            provenance="audit-resolved",
            note="Balanced modeling subset derived from the upstream sample.",
        ),
        CaseStudyNumber(
            key="boolean_strata",
            value=4,
            unit="strata",
            provenance="audit-resolved",
            note="AFSC/UAEORO boolean scenario combinations.",
        ),
        CaseStudyNumber(
            key="rows_per_boolean_stratum",
            value=5_000,
            unit="rows",
            provenance="audit-resolved",
            note="Rows drawn independently within each boolean scenario combination.",
        ),
        CaseStudyNumber(
            key="notebook_train_rows",
            value=18_000,
            unit="rows",
            provenance="notebook-derived",
            note="Notebook train split after the 10 percent external holdout.",
        ),
        CaseStudyNumber(
            key="notebook_holdout_rows",
            value=2_000,
            unit="rows",
            provenance="notebook-derived",
            note="Notebook external holdout split size.",
        ),
        CaseStudyNumber(
            key="candidate_input_count",
            value=352,
            unit="inputs",
            provenance="notebook-derived",
            note="Expanded feature count loaded by the notebook at model start.",
        ),
        CaseStudyNumber(
            key="full_output_count",
            value=23_495,
            unit="outputs",
            provenance="notebook-derived",
            note="Total outputs before notebook output culling.",
        ),
        CaseStudyNumber(
            key="culled_output_count",
            value=9_782,
            unit="outputs",
            provenance="notebook-derived",
            note="Outputs retained for the notebook PCA-LASSO search.",
        ),
        CaseStudyNumber(
            key="selected_feature_count",
            value=346,
            unit="features",
            provenance="notebook-derived",
            note="Retained feature count after the notebook screening stage.",
        ),
        CaseStudyNumber(
            key="first_order_selected_features",
            value=62,
            unit="features",
            provenance="notebook-derived",
            note="First-order terms in the recovered selected-feature summary.",
        ),
        CaseStudyNumber(
            key="nonlinear_selected_features",
            value=40,
            unit="features",
            provenance="notebook-derived",
            note="Nonlinear transformations in the recovered selected-feature summary.",
        ),
        CaseStudyNumber(
            key="second_order_selected_features",
            value=244,
            unit="features",
            provenance="notebook-derived",
            note="Second-order interaction terms in the recovered selected-feature summary.",
        ),
        CaseStudyNumber(
            key="null_permutation_count",
            value=200,
            unit="replicates",
            provenance="source-derived",
            note="Permutation-null count used in null_distribution.py.",
        ),
        CaseStudyNumber(
            key="null_resample_size",
            value=2_000,
            unit="rows",
            provenance="source-derived",
            note="Influential-resampling size used after upstream screening.",
        ),
    )


def case_study_number_table() -> pd.DataFrame:
    """Return recovered case-study numbers as a table.

    Returns
    -------
    pandas.DataFrame
        One row per recovered quantity ordered as documented by the workflow audit.
    """
    return pd.DataFrame(asdict(item) for item in canonical_case_study_numbers())


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
        Original-order candidate input list before screening. Defaults to ``X_train``
        column order.
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
    ValueError
        Raised when screening retains no features.
    """
    resolved_all_features = list(all_input_features or [str(column) for column in X_train.columns])
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
        :func:`bsm_rfm.final_ols.build_postfit_artifacts`.
    root
        Destination bundle directory.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping from logical artifact names to the written on-disk paths.
    """
    bundle_root = Path(root)
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
    destination = root / relative_path
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
