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
    FeatureExpansionResult,
    FeatureExpansionSpec,
    apply_feature_expansion,
    feature_catalog_from_spec,
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
    CaseStudyNumber,
    WorkflowStage,
    canonical_case_study_numbers,
    canonical_workflow_stages,
    case_study_number_table,
    workflow_stage_table,
)

__all__ = [
    "CaseStudyNumber",
    "FeatureExpansionResult",
    "FeatureExpansionSpec",
    "KNOWN_TRANSFORMATIONS",
    "NullScreeningConfig",
    "NullScreeningResult",
    "PipelineManifest",
    "StandardizationBundle",
    "WorkflowStage",
    "add_scenario_flags",
    "align_xy",
    "apply_feature_expansion",
    "bootstrap_macro_nrmse_ci",
    "build_position_map",
    "canonical_case_study_numbers",
    "canonical_module_from_factor_name",
    "canonical_workflow_stages",
    "case_study_number_table",
    "ensure_id_columns",
    "feature_catalog_from_spec",
    "fit_standardizers",
    "load_source_module",
    "make_boolean_combination_labels",
    "make_feature_expansion_spec",
    "macro_nrmse_with_ref",
    "make_metadata_frame",
    "make_null_mean_prediction",
    "parse_selected_input_structure",
    "run_null_screening_with_source",
    "stratified_holdout_split",
    "stratified_subset_by_boolean_combination",
    "workflow_stage_table",
]
