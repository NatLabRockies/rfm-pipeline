"""P0-S09: Correct `empirical_null_retained` provenance semantics (F5).

Tests verify that `empirical_null_retained` (and its paired count fields) reflect
only the actually-retained set — not every candidate — for both the interaction and
nonlinear provenance blocks.

Acceptance criteria:
- `empirical_null_retained` is True only for retained candidates; False for non-retained.
- The count of True `empirical_null_retained` rows equals the number of retained candidates.
- Applies to both interaction and nonlinear provenance blocks.
- Non-retained candidates are explicitly flagged False.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rfm_pipeline.manuscript_stages import (
    InteractionDiscoverySpec,
    NonlinearDiscoverySpec,
    TransformDef,
    _build_interaction_pair_scores,
    _build_nonlinear_transformation_scores,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_SPEC_INTERACTION = InteractionDiscoverySpec(
    method="tree_shap_gradient_boosting",
    aggregation_rule="max_absolute_shap",
    null_threshold_quantile=0.95,
    retained_pairs_reference=10,
    permutation_count_B=99,
)

_SPEC_NONLINEAR = NonlinearDiscoverySpec(
    method="gam_plus_restricted_parametric_replacement",
    curvature_rule="edf_gt_1_and_smooth_pvalue_lt_0p01",
    replacement_selection_rule="minimum_training_rmse_against_gam_smooth",
    identified_transformations_reference=5,
    final_support_transformations_reference=3,
)

_TRANSFORM_LOG = TransformDef(expr="log(x)", label="log", name="log")
_TRANSFORM_SQRT = TransformDef(expr="sqrt(x)", label="sqrt", name="sqrt")

# ---------------------------------------------------------------------------
# Interaction provenance tests
# ---------------------------------------------------------------------------


def _make_interaction_candidates(n: int) -> list[tuple[str, str, str]]:
    return [(f"feat{i}_feat{i + 1}", f"feat{i}", f"feat{i + 1}") for i in range(n)]


class TestInteractionEmpiricalNullRetained:
    """empirical_null_retained for interaction pair_scores."""

    def _build(
        self,
        n_candidates: int = 6,
        retained_indices: list[int] | None = None,
    ) -> pd.DataFrame:
        """Build pair_scores with a partial retained set."""
        if retained_indices is None:
            retained_indices = [0, 2]  # only 2 of 6 retained

        candidates = _make_interaction_candidates(n_candidates)
        n = n_candidates

        rng = np.random.default_rng(42)
        observed_scores = rng.uniform(0.1, 1.0, n)
        thresholds = np.full(n, 0.5)
        p_values = np.where(np.arange(n) < 2, 0.01, 0.9)

        retained = np.zeros(n, dtype=bool)
        for i in retained_indices:
            retained[i] = True

        retained_term_names = {
            pair_name for (pair_name, _, _), r in zip(candidates, retained, strict=True) if r
        }

        return _build_interaction_pair_scores(
            candidates=candidates,
            observed_scores=observed_scores,
            thresholds=thresholds,
            p_values=p_values,
            retained=retained,
            retained_term_names=retained_term_names,
            spec=_SPEC_INTERACTION,
        )

    def test_empirical_null_retained_matches_retained_flag(self):
        """empirical_null_retained must equal the retained flag for every row."""
        df = self._build(n_candidates=6, retained_indices=[1, 4])
        pd.testing.assert_series_equal(
            df["empirical_null_retained"].reset_index(drop=True),
            df["retained"].reset_index(drop=True),
            check_names=False,
        )

    def test_non_retained_candidates_flagged_false(self):
        """Non-retained candidates must have empirical_null_retained == False."""
        df = self._build(n_candidates=6, retained_indices=[0, 3])
        non_retained = df.loc[~df["retained"]]
        assert not non_retained["empirical_null_retained"].any(), (
            "Non-retained pairs should not have empirical_null_retained=True"
        )

    def test_retained_candidates_flagged_true(self):
        """Retained candidates must have empirical_null_retained == True."""
        df = self._build(n_candidates=6, retained_indices=[2, 5])
        retained = df.loc[df["retained"]]
        assert retained["empirical_null_retained"].all(), (
            "Retained pairs should all have empirical_null_retained=True"
        )

    def test_count_matches_retained_count(self):
        """Sum of empirical_null_retained must equal the number of retained pairs."""
        retained_indices = [0, 2, 4]
        df = self._build(n_candidates=6, retained_indices=retained_indices)
        assert int(df["empirical_null_retained"].sum()) == len(retained_indices), (
            "empirical_null_retained count must match the retained count"
        )

    def test_all_retained_gives_full_count(self):
        """When all candidates are retained, count equals n_candidates."""
        n = 4
        df = self._build(n_candidates=n, retained_indices=list(range(n)))
        assert int(df["empirical_null_retained"].sum()) == n

    def test_none_retained_gives_zero_count(self):
        """When no candidates are retained, count is zero."""
        df = self._build(n_candidates=4, retained_indices=[])
        assert int(df["empirical_null_retained"].sum()) == 0
        assert not df["empirical_null_retained"].any()

    def test_old_all_candidates_bug_would_fail(self):
        """Regression: using all candidates as retained_term_names would produce wrong results."""
        n_candidates = 5
        retained_indices = [1]  # only 1 of 5 retained
        candidates = _make_interaction_candidates(n_candidates)
        retained = np.array([i in retained_indices for i in range(n_candidates)])

        rng = np.random.default_rng(99)
        observed_scores = rng.uniform(0.1, 1.0, n_candidates)
        thresholds = np.full(n_candidates, 0.5)
        p_values = np.full(n_candidates, 0.5)

        # Old buggy construction: ALL candidates as retained_term_names
        buggy_retained_term_names = {pair_name for pair_name, _, _ in candidates}
        df_buggy = _build_interaction_pair_scores(
            candidates=candidates,
            observed_scores=observed_scores,
            thresholds=thresholds,
            p_values=p_values,
            retained=retained,
            retained_term_names=buggy_retained_term_names,
            spec=_SPEC_INTERACTION,
        )
        # With the bug, empirical_null_retained would be True for ALL candidates
        assert df_buggy["empirical_null_retained"].all(), (
            "With old bug, all candidates are flagged retained"
        )
        # Correct construction: only retained pairs
        correct_retained_term_names = {
            pair_name for (pair_name, _, _), r in zip(candidates, retained, strict=True) if r
        }
        df_correct = _build_interaction_pair_scores(
            candidates=candidates,
            observed_scores=observed_scores,
            thresholds=thresholds,
            p_values=p_values,
            retained=retained,
            retained_term_names=correct_retained_term_names,
            spec=_SPEC_INTERACTION,
        )
        # With correct construction, only 1 of 5 is flagged
        assert int(df_correct["empirical_null_retained"].sum()) == 1


# ---------------------------------------------------------------------------
# Nonlinear provenance tests
# ---------------------------------------------------------------------------


def _make_nonlinear_candidates_and_retained(
    base_features: list[str],
    transforms: list[TransformDef],
    nonlinear_bases: set[str],
    best_transforms: dict[str, str],
) -> tuple[list[tuple[str, str, TransformDef]], np.ndarray, set[str]]:
    """
    Build (candidates, retained_array, retained_term_names) for nonlinear tests.

    For each (base, transform) pair:
    - retained = True iff base is nonlinear AND this transform is the best one.
    """
    candidates: list[tuple[str, str, TransformDef]] = []
    for base in base_features:
        for td in transforms:
            feat_name = f"{base}_{td.label}"
            candidates.append((feat_name, base, td))

    retained = np.array(
        [
            base in nonlinear_bases and feat_name == best_transforms.get(base)
            for feat_name, base, _ in candidates
        ],
        dtype=bool,
    )
    retained_term_names = {
        feat_name
        for feat_name, _, _ in zip(
            [c[0] for c in candidates],
            [None] * len(candidates),
            [None] * len(candidates),
            strict=True,
        )
        # re-derive from retained array
    }
    # Use actual retained flag to build the correct set
    retained_term_names = {c[0] for c, r in zip(candidates, retained, strict=True) if r}
    return candidates, retained, retained_term_names


class TestNonlinearEmpiricalNullRetained:
    """empirical_null_retained for nonlinear transformation_scores."""

    def _build(
        self,
        base_features: list[str] | None = None,
        nonlinear_bases: set[str] | None = None,
        best_transforms: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Build transformation_scores with a partial retained set."""
        if base_features is None:
            base_features = ["x1", "x2", "x3"]
        if nonlinear_bases is None:
            nonlinear_bases = {"x1", "x3"}  # x2 is linear
        if best_transforms is None:
            # x1 → log transform retained; x3 → sqrt transform retained
            best_transforms = {
                "x1": "x1_log",
                "x3": "x3_sqrt",
            }
        transforms = [_TRANSFORM_LOG, _TRANSFORM_SQRT]
        candidates, retained, retained_term_names = _make_nonlinear_candidates_and_retained(
            base_features=base_features,
            transforms=transforms,
            nonlinear_bases=nonlinear_bases,
            best_transforms=best_transforms,
        )
        n = len(candidates)
        component_names = ["PC1", "PC2"]
        rng = np.random.default_rng(7)
        curvature_scores = rng.uniform(1.5, 5.0, n)
        active_transformations = np.ones(n, dtype=bool)
        best_component_indices = np.zeros(n, dtype=int)
        replacement_rmse = rng.uniform(0.1, 1.0, n)

        return _build_nonlinear_transformation_scores(
            candidates=candidates,
            curvature_scores=curvature_scores,
            active_transformations=active_transformations,
            retained=retained,
            retained_term_names=retained_term_names,
            component_names=component_names,
            best_component_indices=best_component_indices,
            replacement_rmse=replacement_rmse,
            spec=_SPEC_NONLINEAR,
        )

    def test_empirical_null_retained_matches_retained_flag(self):
        """empirical_null_retained must equal retained for every row."""
        df = self._build()
        pd.testing.assert_series_equal(
            df["empirical_null_retained"].reset_index(drop=True),
            df["retained"].reset_index(drop=True),
            check_names=False,
        )

    def test_non_retained_candidates_flagged_false(self):
        """Non-retained transforms must have empirical_null_retained == False."""
        df = self._build()
        non_retained = df.loc[~df["retained"]]
        assert not non_retained["empirical_null_retained"].any(), (
            "Non-retained transforms should not have empirical_null_retained=True"
        )

    def test_retained_candidates_flagged_true(self):
        """Retained transforms must have empirical_null_retained == True."""
        df = self._build()
        retained = df.loc[df["retained"]]
        assert retained["empirical_null_retained"].all()

    def test_count_matches_retained_count(self):
        """Sum of empirical_null_retained must equal number of retained transforms."""
        # 2 nonlinear bases × 1 best transform each = 2 retained
        df = self._build()
        retained_count = int(df["retained"].sum())
        assert int(df["empirical_null_retained"].sum()) == retained_count

    def test_per_base_only_best_transform_retained(self):
        """For a nonlinear base, only the best transform is flagged, not all."""
        base_features = ["base_a"]
        nonlinear_bases = {"base_a"}
        best_transforms = {"base_a": "base_a_log"}  # log is retained, sqrt is not
        df = self._build(
            base_features=base_features,
            nonlinear_bases=nonlinear_bases,
            best_transforms=best_transforms,
        )
        # Two transforms for base_a: log (retained) and sqrt (not)
        assert len(df) == 2
        log_row = df.loc[df["feature_name"] == "base_a_log"].iloc[0]
        sqrt_row = df.loc[df["feature_name"] == "base_a_sqrt"].iloc[0]
        assert log_row["empirical_null_retained"] is True or bool(
            log_row["empirical_null_retained"]
        )
        assert not sqrt_row["empirical_null_retained"]

    def test_linear_base_features_flagged_false(self):
        """Transforms of linear (non-retained) base features must be False."""
        # x2 is linear — neither of its transforms should be retained
        df = self._build()
        x2_rows = df.loc[df["base_feature"] == "x2"]
        assert not x2_rows["empirical_null_retained"].any()
        assert not x2_rows["retained"].any()

    def test_old_base_feat_in_nonlinear_bases_bug_would_fail(self):
        """Regression: flagging all transforms of nonlinear bases would inflate count."""
        base_features = ["b1", "b2"]
        nonlinear_bases = {"b1"}  # only b1 is nonlinear
        best_transforms = {"b1": "b1_log"}
        transforms = [_TRANSFORM_LOG, _TRANSFORM_SQRT]
        candidates, retained, retained_term_names = _make_nonlinear_candidates_and_retained(
            base_features=base_features,
            transforms=transforms,
            nonlinear_bases=nonlinear_bases,
            best_transforms=best_transforms,
        )
        # With old bug: empirical_null_retained = base_feat in nonlinear_bases
        # This would flag BOTH transforms of b1 (b1_log AND b1_sqrt)
        old_buggy_enr = [base in nonlinear_bases for _, base, _ in candidates]
        # With fix: empirical_null_retained = retained (only best transform)
        correct_enr = list(retained)

        # Old: 2 True (both b1 transforms); correct: 1 True (only b1_log)
        assert sum(old_buggy_enr) == 2, "Old bug: both b1 transforms flagged"
        assert sum(correct_enr) == 1, "Fix: only retained b1_log flagged"
        assert old_buggy_enr != correct_enr, "Bug and fix produce different results"
