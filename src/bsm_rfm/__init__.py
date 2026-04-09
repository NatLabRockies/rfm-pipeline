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
from .features import (
    KNOWN_TRANSFORMATIONS,
    canonical_module_from_factor_name,
    parse_selected_input_structure,
)
from .final_ols import (
    canonical_postfit_artifact_names,
    make_coefficient_matrix_frame,
    make_standardization_frame,
    notebook_final_ols_contract,
    postfit_artifact_table,
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

__all__ = [
    "KNOWN_TRANSFORMATIONS",
    "PipelineManifest",
    "StandardizationBundle",
    "add_scenario_flags",
    "align_xy",
    "bootstrap_macro_nrmse_ci",
    "build_position_map",
    "canonical_module_from_factor_name",
    "ensure_id_columns",
    "fit_standardizers",
    "canonical_postfit_artifact_names",
    "load_source_module",
    "make_coefficient_matrix_frame",
    "make_standardization_frame",
    "make_boolean_combination_labels",
    "notebook_final_ols_contract",
    "macro_nrmse_with_ref",
    "make_metadata_frame",
    "make_null_mean_prediction",
    "NullScreeningConfig",
    "NullScreeningResult",
    "parse_selected_input_structure",
    "postfit_artifact_table",
    "run_null_screening_with_source",
    "stratified_holdout_split",
    "stratified_subset_by_boolean_combination",
]
