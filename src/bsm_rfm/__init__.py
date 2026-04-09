"""Foundational utilities for the BSM reduced-form modeling workflow.

The package currently provides small, tested building blocks used to stabilize the
refactor boundary before the full modeling workflow is ported into canonical modules.
"""

from .artifacts import PipelineManifest, build_position_map, make_metadata_frame
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
    SUPPORTED_TRANSFORMS,
    FeatureExpansionResult,
    FeatureExpansionSpec,
    apply_feature_expansion,
    expanded_feature_names,
    make_feature_expansion_spec,
)
from .features import (
    KNOWN_TRANSFORMATIONS,
    canonical_module_from_factor_name,
    parse_selected_input_structure,
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
from .workflow import (
    WorkflowStage,
    canonical_case_study_numbers,
    canonical_workflow_stages,
    case_study_number_table,
    workflow_stage_table,
)

__all__ = [
    "FeatureExpansionResult",
    "FeatureExpansionSpec",
    "KNOWN_TRANSFORMATIONS",
    "SUPPORTED_TRANSFORMS",
    "WorkflowStage",
    "PipelineManifest",
    "apply_feature_expansion",
    "StandardizationBundle",
    "add_scenario_flags",
    "align_xy",
    "bootstrap_macro_nrmse_ci",
    "canonical_case_study_numbers",
    "canonical_workflow_stages",
    "case_study_number_table",
    "build_position_map",
    "canonical_module_from_factor_name",
    "expanded_feature_names",
    "ensure_id_columns",
    "fit_standardizers",
    "make_feature_expansion_spec",
    "load_source_module",
    "make_boolean_combination_labels",
    "macro_nrmse_with_ref",
    "make_metadata_frame",
    "make_null_mean_prediction",
    "NullScreeningConfig",
    "NullScreeningResult",
    "parse_selected_input_structure",
    "run_null_screening_with_source",
    "stratified_holdout_split",
    "workflow_stage_table",
    "stratified_subset_by_boolean_combination",
]
