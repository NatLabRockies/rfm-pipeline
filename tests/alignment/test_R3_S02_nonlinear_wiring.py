"""R3-S02: Wire nonlinear multiplicity correction + fix min-p tracking into production.

The F5 defect: ``discover_manuscript_nonlinear_transformations`` used to declare
per-(feature, component) nonlinearity via an uncorrected raw p<0.01 cut and
selected the replacement transform by minimum training RMSE against the GAM
smooth on the same training data. Additionally, the min-p tracking loop advanced
``best_edf`` and ``best_p`` jointly under a disjunctive condition so ``best_p``
was not guaranteed to be the minimum p-value across active components.

This slice routes the production nonlinear stage through the
multiplicity-corrected discovery rule (Bonferroni over the full input × active-
component search family) and selects the replacement transform family by
independent k-fold cross-validation. It also fixes the min-p tracking bug so
``best_p`` is the true minimum across active components independently of the
EDF-based best-component selection.

These tests exercise the PRODUCTION entry point
:func:`rfm_pipeline.manuscript_stages.discover_manuscript_nonlinear_transformations`
on controlled synthetic fixtures (pure null, planted-log-signal, and a crafted
min-p case with monkeypatched GAM output; fixed seed) and assert:

- family-wise error control: on the same statistics, the corrected multiplicity
  rule yields materially fewer selections than the uncorrected per-component
  0.01 threshold (surfaced via the ``gam_p_value`` column against the raw cut).
- planted-transform recovery: a strongly-planted log-transform signal is
  retained and the log transform family is selected by the CV-based choice.
- min-p correctness: on a crafted 2-component case where EDF is largest on one
  component but the minimum p-value is on the other, the reported ``best_p``
  (surfaced via ``transformation_scores.gam_p_value``) equals the true minimum
  across active components.

Grep guard: ``nonlinear_multiplicity_corrected_alpha`` and
``choose_transform_by_cv`` must be referenced from inside the production module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rfm_pipeline.manuscript_stages as manuscript_stages
from rfm_pipeline.manuscript_stages import (
    NonlinearDiscoverySpec,
    TransformDef,
    discover_manuscript_nonlinear_transformations,
)
from rfm_pipeline.transforms import LOGARITHMIC, SQRT

# ---------------------------------------------------------------------------
# Shared constants and fixture builders
# ---------------------------------------------------------------------------

_SEED = 20260718
_N_SAMPLES = 60
_N_TRAIN = 48


def _make_spec(
    *,
    transform_library: list[TransformDef] | None = None,
    family_wise_alpha: float = 0.05,
    n_cv_splits: int = 5,
) -> NonlinearDiscoverySpec:
    """Build a NonlinearDiscoverySpec with R3-S02 defaults."""
    return NonlinearDiscoverySpec(
        method="gam_plus_restricted_parametric_replacement",
        curvature_rule="edf_gt_1_and_smooth_pvalue_lt_0p01",
        replacement_selection_rule="minimum_training_rmse_against_gam_smooth",
        identified_transformations_reference=0,
        final_support_transformations_reference=0,
        transform_library=(
            transform_library if transform_library is not None else [LOGARITHMIC, SQRT]
        ),
        family_wise_alpha=family_wise_alpha,
        n_cv_splits=n_cv_splits,
        cv_seed=_SEED,
        n_jobs=1,
    )


def _base_tables(
    x_by_feature: dict[str, np.ndarray],
    y_by_component: dict[str, np.ndarray],
    *,
    n_train: int = _N_TRAIN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assemble inputs/catalog/holdout/pca_scores/retained_terms tables.

    ``x_by_feature`` maps input feature name to a length-N array of positive values
    (required for log/sqrt validity). ``y_by_component`` maps PCA component name to
    a length-N array. All arrays must share the same length; the first ``n_train``
    rows are assigned to the training split.
    """
    n_samples = next(iter(x_by_feature.values())).shape[0]
    sample_ids = list(range(1, n_samples + 1))
    inputs = pd.DataFrame({"sample_id": sample_ids})
    for name, values in x_by_feature.items():
        inputs[name] = values
    catalog = pd.DataFrame(
        {
            "feature_name": list(x_by_feature.keys()),
            "feature_type": ["first_order"] * len(x_by_feature),
            "origin": ["test"] * len(x_by_feature),
        }
    )
    holdout = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "split": ["train"] * n_train + ["holdout"] * (n_samples - n_train),
        }
    )
    pca_scores = pd.DataFrame({"sample_id": sample_ids})
    for name, values in y_by_component.items():
        pca_scores[name] = values
    retained_terms = pd.DataFrame(
        {
            "feature_name": list(x_by_feature.keys()),
            "feature_type": ["first_order"] * len(x_by_feature),
        }
    )
    return inputs, catalog, holdout, pca_scores, retained_terms


def _std(v: np.ndarray) -> np.ndarray:
    mean, scale = float(v.mean()), float(v.std(ddof=0))
    if scale <= 0.0:
        return np.zeros_like(v, dtype=float)
    return (v - mean) / scale


# ---------------------------------------------------------------------------
# R3-S02 grep guard: production module must reference the corrected helpers.
# ---------------------------------------------------------------------------


def test_R3_S02_production_module_references_corrected_helpers() -> None:
    """The production module must call both R3-S02 helpers from inside the
    production discovery function (not only define them)."""
    module_path = Path(manuscript_stages.__file__)
    source = module_path.read_text(encoding="utf-8")
    # ``nonlinear_multiplicity_corrected_alpha`` must appear at least twice
    # (definition + at least one call site in production).
    assert source.count("nonlinear_multiplicity_corrected_alpha") >= 2, (
        "discover_manuscript_nonlinear_transformations must call "
        "nonlinear_multiplicity_corrected_alpha; found only the definition."
    )
    # ``choose_transform_by_cv`` must appear at least twice as well (definition
    # + at least one production call site).
    assert source.count("choose_transform_by_cv") >= 2, (
        "discover_manuscript_nonlinear_transformations must call "
        "choose_transform_by_cv; found only the definition."
    )


# ---------------------------------------------------------------------------
# Pure null: false-selection control under the corrected rule
# ---------------------------------------------------------------------------


def test_R3_S02_pure_null_zero_retained_under_corrected_rule() -> None:
    """Under a complete null (all features independent of all components), the
    corrected production path retains zero base features.

    The uncorrected raw p<0.01 threshold on 6 features × 3 components = 18 tests
    has expected false-selection count ~0.18 per run per feature; with a seed
    where at least one component's spline fit has p<0.01 the raw rule would
    retain the feature, but the corrected alpha
    (0.05 / (6*3) = 0.00277...) is strictly smaller so those raw flags are
    filtered out.
    """
    rng = np.random.default_rng(_SEED)
    n_features, n_components = 6, 3
    x_by_feature = {
        f"x{i}": np.abs(rng.standard_normal(_N_SAMPLES)) + 0.1 for i in range(n_features)
    }
    y_by_component = {f"PC{j + 1}": rng.standard_normal(_N_SAMPLES) for j in range(n_components)}

    inputs, catalog, holdout, pca_scores, retained_terms = _base_tables(
        x_by_feature, y_by_component
    )
    spec = _make_spec()

    result = discover_manuscript_nonlinear_transformations(
        inputs, catalog, holdout, pca_scores, retained_terms, spec
    )

    n_retained = int(result.transformation_scores["retained"].sum())
    assert n_retained == 0, (
        f"Under a pure null the corrected production rule should retain 0 "
        f"transformations; got {n_retained}."
    )

    # Verify that the corrected alpha is strictly smaller than the raw 0.01 cut.
    corrected = 0.05 / (n_features * n_components)
    assert corrected < 0.01, (
        f"Sanity: corrected alpha {corrected:.6g} must be < raw 0.01 threshold."
    )

    # The empirical_null_retained diagnostic reflects retention under the
    # (corrected) production rule; assert it agrees with retained.
    assert int(result.transformation_scores["empirical_null_retained"].sum()) == 0


def test_R3_S02_pure_null_strictly_fewer_selections_than_uncorrected() -> None:
    """The corrected production rule yields materially fewer selections than the
    uncorrected raw per-component p<0.01 threshold applied to the same statistics.

    Crafted case: 6 features × 3 components; monkeypatch _gam_test_and_smooth so
    that every feature has at least one component with p just below 0.01
    (uncorrected raw threshold), but none has any component with p below the
    corrected alpha (0.05 / 18 ≈ 0.00278). Under the raw rule, all 6 features
    would be retained; under the corrected rule, zero features are retained.
    """
    rng = np.random.default_rng(_SEED + 1)
    n_features, n_components = 6, 3
    x_by_feature = {
        f"x{i}": np.abs(rng.standard_normal(_N_SAMPLES)) + 0.1 for i in range(n_features)
    }
    y_by_component = {f"PC{j + 1}": rng.standard_normal(_N_SAMPLES) for j in range(n_components)}
    inputs, catalog, holdout, pca_scores, retained_terms = _base_tables(
        x_by_feature, y_by_component
    )
    spec = _make_spec(transform_library=[LOGARITHMIC, SQRT])

    # Craft GAM output: for each x feature, one component has (edf=3.0, p=0.008)
    # which passes the raw 0.01 cut but fails the corrected 0.05/18 cut. The
    # other components return (edf=2.0, p=0.5) — non-significant.
    def _fake_gam(
        x_train: np.ndarray,  # noqa: ARG001
        y_train: np.ndarray,  # noqa: ARG001
        k: int = 3,  # noqa: ARG001
    ) -> tuple[float, float, np.ndarray | None]:
        # Deterministic per-call sequence using a call counter.
        idx = _fake_gam._counter  # type: ignore[attr-defined]
        _fake_gam._counter = idx + 1  # type: ignore[attr-defined]
        # 3 components per feature; make the first one "just below" 0.01.
        if idx % n_components == 0:
            smooth = np.zeros(len(x_train))
            return 3.0, 0.008, smooth
        return 2.0, 0.5, np.zeros(len(x_train))

    _fake_gam._counter = 0  # type: ignore[attr-defined]

    corrected = 0.05 / (n_features * n_components)
    assert 0.008 > corrected, "Fixture invariant: 0.008 must fail the corrected cut."
    assert 0.008 < 0.01, "Fixture invariant: 0.008 must pass the raw 0.01 cut."

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(manuscript_stages, "_gam_test_and_smooth", _fake_gam)
        result = discover_manuscript_nonlinear_transformations(
            inputs, catalog, holdout, pca_scores, retained_terms, spec
        )

    n_retained = int(result.transformation_scores["retained"].sum())
    # Under the raw uncorrected rule, all 6 features would have been retained.
    # Under the corrected rule, zero are retained (0.008 > corrected alpha).
    assert n_retained == 0, (
        f"Corrected production rule must retain 0 features when every feature's "
        f"min p-value fails the corrected cut; got {n_retained}."
    )
    # Sanity: the reported best_p on every base feature equals 0.008 (the
    # crafted minimum across components), confirming min-p tracking works.
    # The transformation_scores table has one row per (feature, transform); all
    # rows for a given base feature share the same gam_p_value.
    unique_ps = result.transformation_scores.groupby("base_feature")["gam_p_value"].nunique().max()
    assert unique_ps == 1, "All transforms for a base feature must share gam_p_value."
    per_base_p = result.transformation_scores.drop_duplicates("base_feature")[
        "gam_p_value"
    ].to_numpy()
    assert np.allclose(per_base_p, 0.008), (
        f"Reported best_p must equal crafted minimum 0.008; got {per_base_p}."
    )


# ---------------------------------------------------------------------------
# Planted transform recovery under corrected rule + CV choice
# ---------------------------------------------------------------------------


def test_R3_S02_planted_log_transform_recovered_by_cv() -> None:
    """A strong planted log-transform signal is retained under the corrected
    multiplicity rule, and CV selects the log transform family over sqrt."""
    rng = np.random.default_rng(_SEED + 2)
    x_signal = rng.uniform(0.5, 5.0, size=_N_SAMPLES)
    log_x = np.log(x_signal)
    y_signal = 6.0 * _std(log_x) + 0.02 * rng.standard_normal(_N_SAMPLES)
    y_noise = rng.standard_normal(_N_SAMPLES)

    x_by_feature = {"x_signal": x_signal}
    y_by_component = {"PC1": y_signal, "PC2": y_noise}
    inputs, catalog, holdout, pca_scores, retained_terms = _base_tables(
        x_by_feature, y_by_component, n_train=_N_TRAIN
    )
    spec = _make_spec(transform_library=[LOGARITHMIC, SQRT])

    result = discover_manuscript_nonlinear_transformations(
        inputs, catalog, holdout, pca_scores, retained_terms, spec
    )

    retained_rows = result.retained_transformations
    assert len(retained_rows) == 1, (
        f"Expected exactly one retained transform (the log family); got "
        f"{len(retained_rows)} rows: {retained_rows.to_dict(orient='records')}."
    )
    retained_family = str(retained_rows.iloc[0]["transformation_family"])
    assert retained_family == "logarithmic", (
        f"Expected CV to select the logarithmic family for the planted log "
        f"signal; got {retained_family!r}."
    )

    # Sanity: corrected alpha bookkeeping should still be strictly < raw 0.01
    # when family size >= 6; here family_size = 1 * 2 = 2, so corrected = 0.025.
    # The test's primary assertion is transform recovery via CV; corrected-alpha
    # inference behavior is exercised in the pure-null tests above.


# ---------------------------------------------------------------------------
# Min-p tracking: reported best_p equals true minimum across components.
# ---------------------------------------------------------------------------


def test_R3_S02_reported_best_p_equals_true_min_across_components() -> None:
    """Crafted 3-component case where EDF is largest on one component but the
    minimum p-value is on a different component. The reported ``best_p``
    (surfaced via ``transformation_scores.gam_p_value``) must equal the true
    minimum p-value across active components — proving the min-p bug is fixed.

    Under the OLD disjunctive rule
    ``if edf > best_edf or (edf > edf_threshold and p < best_p):``
    updating best_p only when the joint condition holds, the returned best_p on
    the crafted sequence (edf, p):
      init:            best_edf=2.0, best_p=1.0
      comp 0: (3.5, 0.5) -> edf>2.0 triggers OR -> best_edf=3.5, best_p=0.5
      comp 1: (1.5, 0.001) -> edf<3.5 and edf<2.0 -> no update
      comp 2: (1.7, 0.5) -> edf<3.5 and edf<2.0 -> no update
    old best_p = 0.5. Correct min-p across components = 0.001.
    """
    rng = np.random.default_rng(_SEED + 3)
    x_by_feature = {"x1": np.abs(rng.standard_normal(_N_SAMPLES)) + 0.1}
    y_by_component = {
        "PC1": rng.standard_normal(_N_SAMPLES),
        "PC2": rng.standard_normal(_N_SAMPLES),
        "PC3": rng.standard_normal(_N_SAMPLES),
    }
    inputs, catalog, holdout, pca_scores, retained_terms = _base_tables(
        x_by_feature, y_by_component, n_train=_N_TRAIN
    )
    spec = _make_spec(transform_library=[LOGARITHMIC, SQRT])

    sequence: list[tuple[float, float]] = [
        (3.5, 0.5),
        (1.5, 0.001),
        (1.7, 0.5),
    ]

    def _fake_gam(
        x_train: np.ndarray,
        y_train: np.ndarray,  # noqa: ARG001
        k: int = 3,  # noqa: ARG001
    ) -> tuple[float, float, np.ndarray | None]:
        idx = _fake_gam._counter  # type: ignore[attr-defined]
        _fake_gam._counter = idx + 1  # type: ignore[attr-defined]
        edf, p = sequence[idx % len(sequence)]
        return float(edf), float(p), np.zeros(len(x_train))

    _fake_gam._counter = 0  # type: ignore[attr-defined]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(manuscript_stages, "_gam_test_and_smooth", _fake_gam)
        result = discover_manuscript_nonlinear_transformations(
            inputs, catalog, holdout, pca_scores, retained_terms, spec
        )

    per_base = (
        result.transformation_scores.drop_duplicates("base_feature")
        .set_index("base_feature")["gam_p_value"]
        .to_dict()
    )
    reported_best_p = float(per_base["x1"])
    true_min_p = min(p for _, p in sequence)

    assert reported_best_p == pytest.approx(true_min_p), (
        f"Reported best_p={reported_best_p!r} must equal true min-p across "
        f"components={true_min_p!r}. The min-p tracking bug is not fixed."
    )
