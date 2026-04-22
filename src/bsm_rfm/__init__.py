"""Public package surface for the BSM reduced-form modeling workflow.

The package exposes tested modules for data preparation, feature expansion, screening,
final OLS fitting, artifact assembly, and workflow orchestration.
"""

from .artifacts import (
    PipelineManifest,
    build_position_map,
    canonical_manifest_position_map_keys,
    canonical_manifest_top_level_keys,
    make_metadata_frame,
)
from .data import (
    StandardizationBundle,
    add_scenario_flags,
    align_xy,
    ensure_id_columns,
    fit_standardizers,
    make_boolean_combination_labels,
    stratified_holdout_split,
    stratified_subset_by_boolean_combination,
)
from .feature_expansion import (
    FeatureExpansionResult,
    FeatureExpansionSpec,
    apply_feature_expansion,
    default_feature_expansion_spec,
    ordered_expanded_feature_names,
)
from .features import (
    KNOWN_TRANSFORMATIONS,
    canonical_module_from_factor_name,
    parse_selected_input_structure,
)
from .final_ols import (
    FinalOLSFitResult,
    build_postfit_artifacts,
    canonical_postfit_artifact_names,
    fit_final_ols,
    make_coefficient_matrix_frame,
    make_holdout_nrmse_summary,
    make_standardization_frame,
    notebook_final_ols_contract,
    postfit_artifact_table,
    predict_final_ols,
)
from .manuscript_data_contract import (
    manuscript_notebook_manifest_table,
    manuscript_notebook_order,
    manuscript_placeholder_path_policy,
    manuscript_required_artifact_table,
    required_manuscript_artifacts,
)
from .metrics import (
    bootstrap_macro_nrmse_ci,
    macro_nrmse_with_ref,
    make_null_mean_prediction,
)
from .null_screening import (
    NullScreeningConfig,
    NullScreeningResult,
    load_source_module,
    run_null_screening_with_source,
)
from .regularized_screening import (
    ScreeningSelectionResult,
    fit_multitask_elastic_net_screen,
    screening_selection_table,
)
from .viz_io import canonical_bundle_loader_keys, load_pipeline_outputs, load_postfit_bundle
from .workflow import (
    CanonicalWorkflowRun,
    canonical_case_study_numbers,
    canonical_workflow_stages,
    case_study_number_table,
    run_canonical_workflow,
    workflow_scope_boundary_table,
    workflow_stage_table,
    write_postfit_bundle,
)

__all__ = [
    "KNOWN_TRANSFORMATIONS",
    "PipelineManifest",
    "StandardizationBundle",
    "add_scenario_flags",
    "align_xy",
    "bootstrap_macro_nrmse_ci",
    "apply_feature_expansion",
    "build_position_map",
    "canonical_bundle_loader_keys",
    "canonical_manifest_position_map_keys",
    "canonical_manifest_top_level_keys",
    "canonical_module_from_factor_name",
    "default_feature_expansion_spec",
    "ensure_id_columns",
    "fit_standardizers",
    "canonical_postfit_artifact_names",
    "FinalOLSFitResult",
    "ScreeningSelectionResult",
    "FeatureExpansionSpec",
    "FeatureExpansionResult",
    "build_postfit_artifacts",
    "fit_final_ols",
    "fit_multitask_elastic_net_screen",
    "load_source_module",
    "load_pipeline_outputs",
    "load_postfit_bundle",
    "make_coefficient_matrix_frame",
    "make_standardization_frame",
    "make_holdout_nrmse_summary",
    "make_boolean_combination_labels",
    "manuscript_notebook_manifest_table",
    "manuscript_notebook_order",
    "manuscript_placeholder_path_policy",
    "manuscript_required_artifact_table",
    "notebook_final_ols_contract",
    "macro_nrmse_with_ref",
    "ordered_expanded_feature_names",
    "make_metadata_frame",
    "make_null_mean_prediction",
    "NullScreeningConfig",
    "NullScreeningResult",
    "parse_selected_input_structure",
    "postfit_artifact_table",
    "predict_final_ols",
    "run_null_screening_with_source",
    "screening_selection_table",
    "stratified_holdout_split",
    "stratified_subset_by_boolean_combination",
    "CanonicalWorkflowRun",
    "canonical_case_study_numbers",
    "canonical_workflow_stages",
    "case_study_number_table",
    "required_manuscript_artifacts",
    "run_canonical_workflow",
    "workflow_scope_boundary_table",
    "workflow_stage_table",
    "write_postfit_bundle",
]
