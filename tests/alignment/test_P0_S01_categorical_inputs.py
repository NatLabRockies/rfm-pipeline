"""P0-S01: Generic categorical/block predictor inputs in config + schema.

Tests:
- valid declaration parses correctly (name-only and name+levels)
- unknown/invalid name raises ValueError
- empty/default preserves existing behavior (categorical_inputs is empty list)
- round-trips: WorkflowConfig constructed directly preserves the field
- malformed levels raise ValueError
"""

from __future__ import annotations

import pytest

from rfm_pipeline.config import (
    CategoricalInputDecl,
    WorkflowConfig,
    _validate_categorical_inputs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_workflow_config(**overrides) -> WorkflowConfig:
    """Return a minimal WorkflowConfig via direct construction (no YAML file)."""
    from rfm_pipeline.config import (
        AlgorithmConfig,
        DatasetConfig,
        OutputConfig,
        RuntimeConfig,
        StagesConfig,
        ValidationConfig,
    )

    return WorkflowConfig(
        dataset=DatasetConfig(type="synthetic_300_sample"),
        algorithm=AlgorithmConfig(),
        runtime=RuntimeConfig(),
        stages=StagesConfig(),
        validation=ValidationConfig(),
        output=OutputConfig(),
        **overrides,
    )


# ---------------------------------------------------------------------------
# Valid declaration parses
# ---------------------------------------------------------------------------


class TestValidDeclarations:
    def test_name_only_string(self):
        decl = CategoricalInputDecl(name="scenario")
        assert decl.name == "scenario"
        assert decl.levels is None

    def test_name_with_levels(self):
        decl = CategoricalInputDecl(name="site", levels=["A", "B", "C"])
        assert decl.name == "site"
        assert decl.levels == ["A", "B", "C"]

    def test_multiple_decls_validate_ok(self):
        decls = [
            CategoricalInputDecl(name="scenario"),
            CategoricalInputDecl(name="treatment_arm", levels=["control", "treated"]),
        ]
        # Must not raise
        _validate_categorical_inputs(decls)

    def test_workflowconfig_accepts_categorical_inputs(self):
        decls = [CategoricalInputDecl(name="block_var")]
        cfg = _minimal_workflow_config(categorical_inputs=decls)
        assert len(cfg.categorical_inputs) == 1
        assert cfg.categorical_inputs[0].name == "block_var"

    def test_underscore_and_mixed_case_names_valid(self):
        decls = [
            CategoricalInputDecl(name="_private"),
            CategoricalInputDecl(name="CamelCase"),
            CategoricalInputDecl(name="var_123"),
        ]
        _validate_categorical_inputs(decls)  # must not raise


# ---------------------------------------------------------------------------
# Unknown / invalid names raise
# ---------------------------------------------------------------------------


class TestInvalidNames:
    @pytest.mark.parametrize(
        "bad_name",
        [
            "",  # empty
            "123bad",  # starts with digit
            "has space",  # contains space
            "a-b",  # hyphen not allowed
            "x.y",  # dot not allowed
        ],
    )
    def test_invalid_name_raises(self, bad_name):
        decls = [CategoricalInputDecl(name=bad_name)]
        with pytest.raises(ValueError, match="categorical_inputs"):
            _validate_categorical_inputs(decls)


# ---------------------------------------------------------------------------
# Malformed levels raise
# ---------------------------------------------------------------------------


class TestMalformedLevels:
    def test_empty_levels_list_raises(self):
        decls = [CategoricalInputDecl(name="scenario", levels=[])]
        with pytest.raises(ValueError, match="non-empty list"):
            _validate_categorical_inputs(decls)

    def test_levels_with_empty_string_raises(self):
        decls = [CategoricalInputDecl(name="scenario", levels=["A", ""])]
        with pytest.raises(ValueError, match="non-empty string"):
            _validate_categorical_inputs(decls)

    def test_levels_with_whitespace_only_raises(self):
        decls = [CategoricalInputDecl(name="scenario", levels=["A", "   "])]
        with pytest.raises(ValueError, match="non-empty string"):
            _validate_categorical_inputs(decls)


# ---------------------------------------------------------------------------
# Empty / default preserves current behavior
# ---------------------------------------------------------------------------


class TestDefaultBehavior:
    def test_default_categorical_inputs_is_empty_list(self):
        cfg = _minimal_workflow_config()
        assert cfg.categorical_inputs == []

    def test_empty_list_validates_without_error(self):
        _validate_categorical_inputs([])  # must not raise

    def test_workflowconfig_without_categorical_inputs_unchanged(self):
        cfg = _minimal_workflow_config()
        # Spot-check that existing fields are unaffected
        assert cfg.dataset.type == "synthetic_300_sample"
        assert cfg.algorithm.variance_threshold == 0.90
        assert cfg.stages.empirical_null_screening.n_permutations == 201


# ---------------------------------------------------------------------------
# Round-trip through direct construction
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_round_trip_preserves_names_and_levels(self):
        decls = [
            CategoricalInputDecl(name="scenario"),
            CategoricalInputDecl(name="site", levels=["north", "south"]),
        ]
        cfg = _minimal_workflow_config(categorical_inputs=decls)
        out = cfg.categorical_inputs
        assert out[0].name == "scenario"
        assert out[0].levels is None
        assert out[1].name == "site"
        assert out[1].levels == ["north", "south"]
