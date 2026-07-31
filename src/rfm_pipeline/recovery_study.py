# Copyright (c) 2026 Dylan Hettinger
"""Prespecified recovery study: manifest, estimands, and empirical FWER.

Provides:
- ``RecoveryScenario`` dataclass carrying a frozen DGP spec, seed, and planted support.
- ``prespecified_recovery_scenarios()`` returning the 7 fixed, seeded scenarios.
- ``recovery_estimands()`` computing per-family and whole-support recovery metrics.
- ``empirical_interaction_fwer()`` estimating interaction FWER from per-replicate counts.

All functions are case-study-agnostic (arbitrary X/Y, no BSM constants).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from .manuscript_stages import (
    EmpiricalNullScreeningSpec,
    InteractionDiscoverySpec,
    NonlinearDiscoverySpec,
    OutputConditioningSpec,
    SparseSelectionStabilitySpec,
    build_manuscript_feature_design,
    condition_manuscript_outputs,
    discover_manuscript_interactions,
    discover_manuscript_nonlinear_transformations,
    screen_manuscript_empirical_null_terms,
    select_manuscript_sparse_support,
)
from .synthetic_dgp import (
    DGPTrueSupport,
    SyntheticDGPSpec,
    generate_calibrated_structure_synthetic,
    generate_pure_synthetic,
)

# Analysis catalog: transforms the method can consider.  A "misspecified" transform
# is one present in the true DGP but absent from this set.
_ANALYSIS_CATALOG_TRANSFORMS: frozenset[str] = frozenset({"sin", "sq", "log1p", "exp"})


# ---------------------------------------------------------------------------
# RecoveryScenario
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecoveryScenario:
    """A prespecified, seeded recovery scenario frozen before any run.

    Attributes
    ----------
    name
        Short scenario identifier (snake_case).
    spec
        Fully parameterized DGP specification including the fixed seed.
    seed
        Mirrors ``spec.seed`` for direct access.
    true_support
        Planted support the method is evaluated against.
    misspecified_transforms
        Feature names (e.g. ``"cubic_x0"``) whose base transform is outside
        ``_ANALYSIS_CATALOG_TRANSFORMS``.  Non-empty only for misspecified scenarios.
    notes
        Human-readable scenario description.
    """

    name: str
    spec: SyntheticDGPSpec
    seed: int
    true_support: DGPTrueSupport
    misspecified_transforms: frozenset[str] = frozenset()
    notes: str = ""


# ---------------------------------------------------------------------------
# Seven prespecified scenarios
# ---------------------------------------------------------------------------


def prespecified_recovery_scenarios() -> list[RecoveryScenario]:
    """Return the 7 prespecified, seeded recovery scenarios frozen before any run.

    Scenarios
    ---------
    1. global_null               — no interactions, no nonlinear transforms.
    2. interaction_null          — main + nonlinear signal; no true interactions.
    3. sparse_strong_hierarchical — sparse main effects with hierarchical interactions.
    4. weak_signal               — per-output SNR heterogeneity (>=2 effective SNR levels).
    5. correlated_redundant      — AR(1)-correlated predictors (rho=0.8).
    6. pure_interaction          — interactions only; main effects declared negligible.
    7. nonlinear_misspecified    — true DGP includes a transform outside the catalog.
    """
    scenarios: list[RecoveryScenario] = []

    # ------------------------------------------------------------------
    # 1. Global null: no planted interactions, no nonlinear terms.
    # ------------------------------------------------------------------
    spec1 = SyntheticDGPSpec(
        n_inputs=20,
        n_runs=400,
        n_outputs=5,
        sparsity=0.10,
        interaction_density=0.0,
        nonlinearity_strength=0.0,
        noise_snr=20.0,
        holdout_fraction=0.10,
        dgp_family="pure_synthetic",
        seed=101,
    )
    ds1 = generate_pure_synthetic(spec1)
    scenarios.append(
        RecoveryScenario(
            name="global_null",
            spec=spec1,
            seed=101,
            true_support=ds1.true_support,
            notes=(
                "No planted interactions or nonlinear transforms; tests interaction FWER "
                "under the complete null for both families."
            ),
        )
    )

    # ------------------------------------------------------------------
    # 2. Interaction-null: main effects + nonlinear; no true interactions.
    # ------------------------------------------------------------------
    spec2 = SyntheticDGPSpec(
        n_inputs=20,
        n_runs=400,
        n_outputs=5,
        sparsity=0.15,
        interaction_density=0.0,
        nonlinearity_strength=0.5,
        noise_snr=20.0,
        holdout_fraction=0.10,
        dgp_family="pure_synthetic",
        seed=102,
    )
    ds2 = generate_pure_synthetic(spec2)
    scenarios.append(
        RecoveryScenario(
            name="interaction_null",
            spec=spec2,
            seed=102,
            true_support=ds2.true_support,
            notes=(
                "Main effects and nonlinear transforms present; no true interactions. "
                "Tests that nonlinear signal does not trigger false interaction discoveries."
            ),
        )
    )

    # ------------------------------------------------------------------
    # 3. Sparse-strong hierarchical: sparse main effects with interactions.
    # ------------------------------------------------------------------
    spec3 = SyntheticDGPSpec(
        n_inputs=30,
        n_runs=400,
        n_outputs=5,
        sparsity=0.10,
        interaction_density=0.35,
        nonlinearity_strength=0.0,
        noise_snr=20.0,
        holdout_fraction=0.10,
        dgp_family="pure_synthetic",
        seed=103,
    )
    ds3 = generate_pure_synthetic(spec3)
    scenarios.append(
        RecoveryScenario(
            name="sparse_strong_hierarchical",
            spec=spec3,
            seed=103,
            true_support=ds3.true_support,
            notes=(
                "Sparse active set with strong hierarchical interactions; "
                "tests power to detect both main and interaction signals."
            ),
        )
    )

    # ------------------------------------------------------------------
    # 4. Weak-signal: per-output SNR heterogeneity creates >=2 SNR levels.
    # ------------------------------------------------------------------
    spec4 = SyntheticDGPSpec(
        n_inputs=20,
        n_runs=400,
        n_outputs=10,
        sparsity=0.15,
        interaction_density=0.2,
        nonlinearity_strength=0.0,
        noise_snr=10.0,
        holdout_fraction=0.10,
        dgp_family="pure_synthetic",
        seed=104,
        per_output_snr_heterogeneity=1.5,
    )
    ds4 = generate_pure_synthetic(spec4)
    scenarios.append(
        RecoveryScenario(
            name="weak_signal",
            spec=spec4,
            seed=104,
            true_support=ds4.true_support,
            notes=(
                "LogNormal(0, 1.5) per-output SNR multiplier creates a wide range of "
                "effective SNR levels (>=2 distinct regimes) across the 10 outputs."
            ),
        )
    )

    # ------------------------------------------------------------------
    # 5. Correlated/redundant predictors: AR(1) rho=0.8.
    # ------------------------------------------------------------------
    spec5 = SyntheticDGPSpec(
        n_inputs=20,
        n_runs=400,
        n_outputs=5,
        sparsity=0.15,
        interaction_density=0.2,
        nonlinearity_strength=0.0,
        noise_snr=15.0,
        holdout_fraction=0.10,
        dgp_family="calibrated_structure",
        seed=105,
        input_correlation_strength=0.8,
    )
    ds5 = generate_calibrated_structure_synthetic(spec5)
    scenarios.append(
        RecoveryScenario(
            name="correlated_redundant",
            spec=spec5,
            seed=105,
            true_support=ds5.true_support,
            notes=(
                "AR(1)-correlated inputs with rho=0.8 create highly redundant predictors; "
                "tests robustness of support recovery under near-collinearity."
            ),
        )
    )

    # ------------------------------------------------------------------
    # 6. Pure-interaction: only interaction effects; main effects negligible.
    # Declared true_active_inputs is empty; interactions are planted.
    # ------------------------------------------------------------------
    spec6 = SyntheticDGPSpec(
        n_inputs=20,
        n_runs=400,
        n_outputs=5,
        sparsity=0.10,
        interaction_density=0.45,
        nonlinearity_strength=0.0,
        noise_snr=20.0,
        holdout_fraction=0.10,
        dgp_family="pure_synthetic",
        seed=106,
    )
    ds6 = generate_pure_synthetic(spec6)
    # Declare main effects as negligible: planted main-effect support is empty.
    pure_interaction_support = DGPTrueSupport(
        true_active_inputs=frozenset(),
        true_active_interactions=ds6.true_support.true_active_interactions,
        true_active_nonlinear=frozenset(),
    )
    scenarios.append(
        RecoveryScenario(
            name="pure_interaction",
            spec=spec6,
            seed=106,
            true_support=pure_interaction_support,
            notes=(
                "Interaction-dominant DGP (interaction_density=0.45). "
                "Planted main-effect support declared empty (~zero main effects); "
                "tests whether the method discovers interactions without main-effect guidance."
            ),
        )
    )

    # ------------------------------------------------------------------
    # 7. Nonlinear + misspecified transform: true DGP includes a cubic
    # transform outside the analysis catalog {sin, sq, log1p, exp}.
    # ------------------------------------------------------------------
    spec7 = SyntheticDGPSpec(
        n_inputs=20,
        n_runs=400,
        n_outputs=5,
        sparsity=0.15,
        interaction_density=0.0,
        nonlinearity_strength=0.6,
        noise_snr=20.0,
        holdout_fraction=0.10,
        dgp_family="pure_synthetic",
        seed=107,
    )
    ds7 = generate_pure_synthetic(spec7)
    # Identify the first active input for the misspecified cubic transform.
    if ds7.true_support.true_active_inputs:
        mismatch_input = min(
            ds7.true_support.true_active_inputs,
            key=lambda s: int(s.removeprefix("x")),
        )
    else:
        mismatch_input = "x0"
    misspecified_name = f"cubic_{mismatch_input}"
    extended_nonlinear = ds7.true_support.true_active_nonlinear | frozenset({misspecified_name})
    misspecified_support = DGPTrueSupport(
        true_active_inputs=ds7.true_support.true_active_inputs,
        true_active_interactions=ds7.true_support.true_active_interactions,
        true_active_nonlinear=extended_nonlinear,
    )
    scenarios.append(
        RecoveryScenario(
            name="nonlinear_misspecified",
            spec=spec7,
            seed=107,
            true_support=misspecified_support,
            misspecified_transforms=frozenset({misspecified_name}),
            notes=(
                f"True DGP includes {misspecified_name} (cubic transform) which is outside "
                "the analysis catalog {sin, sq, log1p, exp}. Exact transformation-family "
                "support recovery is unattainable."
            ),
        )
    )

    return scenarios


# ---------------------------------------------------------------------------
# Recovery estimands
# ---------------------------------------------------------------------------


def recovery_estimands(
    true_support: DGPTrueSupport,
    selected_support: DGPTrueSupport,
) -> dict[str, dict[str, object]]:
    """Compute per-family and whole-support recovery estimands.

    Parameters
    ----------
    true_support
        Planted true support.
    selected_support
        Support selected by the method under evaluation.

    Returns
    -------
    dict
        Nested dict keyed by family (``"main"``, ``"interaction"``,
        ``"transformation"``, ``"whole"``), each mapping to::

            {
                "precision": float,  # TP / selected_size (1.0 if selected is empty)
                "recall": float,  # TP / true_size     (1.0 if true is empty)
                "fdp": float,  # FP / selected_size (0.0 if selected is empty)
                "exact_support_recovery": bool,
                "selected_size": int,
            }

        Interaction-pair metrics use unordered pairs.
    """
    main_metrics = _family_metrics(
        true_support.true_active_inputs,
        selected_support.true_active_inputs,
    )

    # Interaction family: normalise to unordered (frozenset) pairs for comparison.
    true_ix = frozenset(frozenset(p) for p in true_support.true_active_interactions)
    sel_ix = frozenset(frozenset(p) for p in selected_support.true_active_interactions)
    interaction_metrics = _family_metrics(true_ix, sel_ix)

    transform_metrics = _family_metrics(
        true_support.true_active_nonlinear,
        selected_support.true_active_nonlinear,
    )

    # Whole support: union across families using canonical string representations.
    true_whole = (
        frozenset(true_support.true_active_inputs)
        | frozenset(_interaction_str(p) for p in true_support.true_active_interactions)
        | frozenset(true_support.true_active_nonlinear)
    )
    sel_whole = (
        frozenset(selected_support.true_active_inputs)
        | frozenset(_interaction_str(p) for p in selected_support.true_active_interactions)
        | frozenset(selected_support.true_active_nonlinear)
    )
    whole_metrics = _family_metrics(true_whole, sel_whole)

    return {
        "main": main_metrics,
        "interaction": interaction_metrics,
        "transformation": transform_metrics,
        "whole": whole_metrics,
    }


def _interaction_str(pair: tuple[str, str] | frozenset[str]) -> str:
    parts = sorted(pair)
    return f"{parts[0]}:{parts[1]}"


def _family_metrics(
    true_set: frozenset,
    selected_set: frozenset,
) -> dict[str, object]:
    tp = len(true_set & selected_set)
    fp = len(selected_set - true_set)
    selected_size = len(selected_set)
    true_size = len(true_set)
    precision: float = tp / selected_size if selected_size > 0 else 1.0
    recall: float = tp / true_size if true_size > 0 else 1.0
    fdp: float = fp / selected_size if selected_size > 0 else 0.0
    exact_recovery: bool = selected_set == true_set
    return {
        "precision": precision,
        "recall": recall,
        "fdp": fdp,
        "exact_support_recovery": exact_recovery,
        "selected_size": selected_size,
    }


# ---------------------------------------------------------------------------
# Empirical interaction FWER
# ---------------------------------------------------------------------------


def empirical_interaction_fwer(
    false_pair_flags: Sequence[int],
    *,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Estimate the empirical interaction FWER from per-replicate false-pair counts.

    Parameters
    ----------
    false_pair_flags
        Per-replicate count (or bool) of falsely selected interaction pairs.
        FWER is estimated as the proportion of replicates with count > 0.
    confidence
        Confidence level for the Wilson interval (default 0.95).

    Returns
    -------
    dict
        ``fwer_proportion``
            Fraction of replicates with >= 1 false interaction pair.
        ``wilson_ci_lower``, ``wilson_ci_upper``
            Wilson (score) confidence interval for ``fwer_proportion``.
        ``n_replicates``
            Total replicate count.
        ``n_false_pair_replicates``
            Number of replicates with >= 1 false pair (the FWER numerator).
        ``mean_false_pair_count``
            Per-replicate mean count of false pairs.  **Not** an FWER; reported
            separately as a per-comparison rate diagnostic.

    Notes
    -----
    Only ``fwer_proportion`` and its Wilson CI measure FWER.  The
    ``mean_false_pair_count`` is a per-comparison average and must not be
    labelled as FWER.
    """
    counts = np.asarray(false_pair_flags, dtype=np.int64)
    n = int(len(counts))
    if n == 0:
        return {
            "fwer_proportion": 0.0,
            "wilson_ci_lower": 0.0,
            "wilson_ci_upper": 1.0,
            "n_replicates": 0,
            "n_false_pair_replicates": 0,
            "mean_false_pair_count": 0.0,
        }

    k = int((counts > 0).sum())
    proportion = k / n

    ci_lower, ci_upper = _wilson_ci(k=k, n=n, confidence=confidence)

    return {
        "fwer_proportion": proportion,
        "wilson_ci_lower": ci_lower,
        "wilson_ci_upper": ci_upper,
        "n_replicates": n,
        "n_false_pair_replicates": k,
        "mean_false_pair_count": float(counts.mean()),
    }


def _wilson_ci(k: int, n: int, confidence: float) -> tuple[float, float]:
    """Wilson (score) confidence interval for a binomial proportion k/n."""
    z = float(_norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    denom = n + z**2
    center = (k + z**2 / 2.0) / denom
    variance_term = max(0.0, k * (n - k) / n + z**2 / 4.0)
    half = z * sqrt(variance_term) / denom
    return max(0.0, center - half), min(1.0, center + half)


# ---------------------------------------------------------------------------
# ProductionRecoveryResult and run_production_recovery_pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProductionRecoveryResult:
    """Per-stage recovery results from the production pipeline runner.

    Attributes
    ----------
    screening_candidate_count
        Number of candidate first-order terms evaluated in empirical-null screening.
    screening_retained_set
        First-order feature names retained by BH empirical-null screening.
    interaction_candidate_count
        Number of candidate interaction pairs evaluated.
    interaction_retained_set
        Interaction pair names (``"left:right"`` notation) retained by the
        FWER-controlled family-error procedure.
    nonlinear_candidate_count
        Number of candidate (base-feature, transform) combinations evaluated.
    nonlinear_retained_set
        Transform feature names (e.g. ``"x0_sq"``) retained by nonlinear discovery.
    final_selected_support
        Feature/interaction/transformation names surviving sparse EBIC selection
        and stability filtering.
    eval_predictions
        Final-OLS predictions on the eval set; shape ``(n_eval, n_outputs)``.
    """

    screening_candidate_count: int
    screening_retained_set: frozenset[str]
    interaction_candidate_count: int
    interaction_retained_set: frozenset[str]
    nonlinear_candidate_count: int
    nonlinear_retained_set: frozenset[str]
    final_selected_support: frozenset[str]
    eval_predictions: np.ndarray


def run_production_recovery_pipeline(
    X_train: np.ndarray | pd.DataFrame,
    Y_train: np.ndarray | pd.DataFrame,
    X_eval: np.ndarray | pd.DataFrame,
    Y_eval: np.ndarray | pd.DataFrame,
    *,
    n_pca_components: int = 2,
    permutation_count_B: int = 999,
    alpha: float = 0.05,
    family_error_method: str = "fwer_max_stat_exact",
    n_tree_estimators: int = 50,
    seed: int = 0,
) -> ProductionRecoveryResult:
    """Run the production discovery/selection stages on arbitrary in-memory data.

    Exercises the SHIPPED stage functions on arbitrary in-memory ``(X_train,
    Y_train, X_eval, Y_eval)`` at a documented reduced scale, measuring
    recovery estimands and empirical FWER across the four production stages.
    No BSM constants and no ``ManuscriptNotebookContext`` are required.

    Parameters
    ----------
    X_train
        Training inputs; shape ``(n_train, n_inputs)``.  May be a numpy array
        or a pandas DataFrame.  Column names from DataFrames are preserved;
        numpy arrays receive auto-names ``x0, x1, ...``.
    Y_train
        Training responses; shape ``(n_train, n_outputs)`` or ``(n_train,)``.
    X_eval
        Evaluation inputs; shape ``(n_eval, n_inputs)``.  Must share column
        semantics with ``X_train``.
    Y_eval
        Evaluation responses; shape ``(n_eval, n_outputs)`` or ``(n_eval,)``.
        Used only as reference; not seen during any training-side fitting.
    n_pca_components
        Maximum number of PCA components retained from the output-conditioning
        stage.  Default ``2``.
    permutation_count_B
        Number of permutations used by empirical-null screening and interaction
        discovery.  Default ``999`` (minimum draws for resolvable exact FWER
        p-values at alpha=0.05).
    alpha
        Family-wise error rate target for interaction discovery and nonlinear
        discovery.  Default ``0.05``.
    family_error_method
        Multiplicity-correction method for interaction discovery.  Must be one
        of ``"fwer_max_stat"``, ``"fwer_max_stat_exact"``, or ``"bh_fdr"``.
        Default ``"fwer_max_stat_exact"``.
    n_tree_estimators
        Number of gradient-boosted trees used per SHAP interaction score.
        Lower values speed up the runner at the cost of interaction-detection
        power.  Default ``50``.
    seed
        Master random seed; propagated to all permutation and resampling steps
        for full determinism.  Default ``0``.

    Returns
    -------
    ProductionRecoveryResult
        Per-stage candidate counts, retained sets, final selected support, and
        final-OLS predictions on ``X_eval``.

    Notes
    -----
    Stage sequencing
        1. ``condition_manuscript_outputs`` — PCA reduction of ``Y_train``.
        2. ``screen_manuscript_empirical_null_terms`` — BH empirical-null screen.
        3. ``discover_manuscript_interactions`` — FWER-controlled maxT detection.
        4. ``discover_manuscript_nonlinear_transformations`` — GAM curvature screen.
        5. ``select_manuscript_sparse_support`` — EBIC-L1 with stability filtering.

    Graceful degradation
        If a stage cannot run (e.g. fewer than two retained first-order terms
        for interaction discovery), it returns empty retained sets and the
        pipeline continues with the available upstream evidence.
    """

    # ------------------------------------------------------------------
    # 1. Coerce inputs to 2-D float arrays
    # ------------------------------------------------------------------
    def _to_arr(x: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            return x.to_numpy(dtype=float)
        arr = np.asarray(x, dtype=float)
        return arr if arr.ndim == 2 else arr[:, None]

    X_tr_arr = _to_arr(X_train)
    Y_tr_arr = _to_arr(Y_train)
    X_ev_arr = _to_arr(X_eval)

    n_train, n_inputs = X_tr_arr.shape
    n_eval = X_ev_arr.shape[0]
    n_outputs = Y_tr_arr.shape[1]

    # Derive column names (preserve DataFrame column names when available)
    if isinstance(X_train, pd.DataFrame):
        input_names = [str(c) for c in X_train.columns if str(c) != "sample_id"]
    else:
        input_names = [f"x{i}" for i in range(n_inputs)]

    if isinstance(Y_train, pd.DataFrame):
        output_names = [str(c) for c in Y_train.columns if str(c) != "sample_id"]
    else:
        output_names = [f"y{j}" for j in range(n_outputs)]

    train_ids: list[int] = list(range(n_train))
    eval_ids: list[int] = list(range(n_train, n_train + n_eval))
    all_ids = train_ids + eval_ids

    # ------------------------------------------------------------------
    # 2. Build combined DataFrames for the stage functions
    # ------------------------------------------------------------------
    X_ev_arr_coerced = _to_arr(X_eval)
    X_all = np.vstack([X_tr_arr, X_ev_arr_coerced])
    input_matrix = pd.DataFrame(
        {"sample_id": all_ids, **{name: X_all[:, i] for i, name in enumerate(input_names)}}
    )

    Y_ev_arr = _to_arr(Y_eval)
    Y_all = np.vstack([Y_tr_arr, Y_ev_arr])
    output_matrix = pd.DataFrame(
        {"sample_id": all_ids, **{name: Y_all[:, j] for j, name in enumerate(output_names)}}
    )

    holdout_assignments = pd.DataFrame(
        {
            "sample_id": all_ids,
            "split": ["train"] * n_train + ["holdout"] * n_eval,
        }
    )
    feature_catalog = pd.DataFrame(
        {
            "feature_name": input_names,
            "feature_type": ["first_order"] * n_inputs,
        }
    )

    # ------------------------------------------------------------------
    # 3. Build stage specs (no BSM config files; all parameters are generic)
    # ------------------------------------------------------------------
    conditioning_spec = OutputConditioningSpec(
        epsilon_var=0.0,
        epsilon_snr=0.0,
        snr_delta=1.0,
        method="pca",
        retained_components=n_pca_components,
        retained_variance_fraction=0.99,
    )
    screening_spec = EmpiricalNullScreeningSpec(
        statistic="coefficient_row_l2_norm",
        permutation_count_B=permutation_count_B,
        bh_q_screen=0.20,
        retained_terms_reference=0,
        random_seed=seed,
    )
    interaction_spec = InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.95,
        retained_pairs_reference=0,
        permutation_count_B=permutation_count_B,
        random_seed=seed,
        family_error_method=family_error_method,
        family_error_alpha=alpha,
        enforce_permutation_adequacy=False,
        n_jobs=1,
        n_tree_estimators=n_tree_estimators,
        max_shap_samples=n_train,
    )
    nonlinear_spec = NonlinearDiscoverySpec(
        method="gam_plus_restricted_parametric_replacement",
        curvature_rule="edf_gt_1_and_smooth_pvalue_lt_0p01",
        replacement_selection_rule="minimum_training_rmse_against_gam_smooth",
        identified_transformations_reference=0,
        final_support_transformations_reference=0,
        family_wise_alpha=alpha,
        cv_seed=seed,
    )
    sparse_spec = SparseSelectionStabilitySpec(
        model_class="l1_penalized_linear_model_per_retained_component",
        ebic_gamma=0.5,
        subsample_count=5,
        subsample_fraction=0.8,
        jaccard_threshold=0.5,
        spearman_threshold=0.5,
        random_seed=seed,
    )

    # ------------------------------------------------------------------
    # 4. Stage 1: output conditioning + PCA
    # ------------------------------------------------------------------
    conditioning = condition_manuscript_outputs(
        output_matrix, holdout_assignments, conditioning_spec
    )
    pca_scores = conditioning.pca_scores

    # ------------------------------------------------------------------
    # 5. Stage 2: empirical-null screening
    # ------------------------------------------------------------------
    screening = screen_manuscript_empirical_null_terms(
        input_matrix,
        feature_catalog,
        holdout_assignments,
        pca_scores,
        screening_spec,
    )
    retained_terms = screening.retained_terms
    screening_candidate_count = len(screening.feature_screening_statistics)
    screening_retained_set: frozenset[str] = frozenset(retained_terms["feature_name"].astype(str))

    # ------------------------------------------------------------------
    # 6. Stage 3: interaction discovery (requires >= 2 retained first-order terms)
    # ------------------------------------------------------------------
    interaction_candidate_count = 0
    interaction_retained_set: frozenset[str] = frozenset()
    retained_pairs: pd.DataFrame = pd.DataFrame(columns=["pair_name", "feature_name"])
    try:
        ix_result = discover_manuscript_interactions(
            input_matrix,
            feature_catalog,
            holdout_assignments,
            pca_scores,
            retained_terms,
            interaction_spec,
        )
        retained_pairs = ix_result.retained_pairs
        interaction_candidate_count = len(ix_result.pair_scores)
        if not retained_pairs.empty:
            pair_col = "pair_name" if "pair_name" in retained_pairs.columns else "feature_name"
            interaction_retained_set = frozenset(retained_pairs[pair_col].astype(str))
    except ValueError:
        pass

    # ------------------------------------------------------------------
    # 7. Stage 4: nonlinear discovery
    # ------------------------------------------------------------------
    nonlinear_candidate_count = 0
    nonlinear_retained_set: frozenset[str] = frozenset()
    retained_transformations: pd.DataFrame = pd.DataFrame(columns=["feature_name"])
    try:
        nl_result = discover_manuscript_nonlinear_transformations(
            input_matrix,
            feature_catalog,
            holdout_assignments,
            pca_scores,
            retained_terms,
            nonlinear_spec,
        )
        retained_transformations = nl_result.retained_transformations
        nonlinear_candidate_count = len(nl_result.transformation_scores)
        if not retained_transformations.empty:
            nonlinear_retained_set = frozenset(retained_transformations["feature_name"].astype(str))
    except ValueError:
        pass

    # ------------------------------------------------------------------
    # 8. Stage 5: sparse EBIC selection + stability filtering
    # ------------------------------------------------------------------
    final_selected_support: frozenset[str] = frozenset()
    try:
        sparse_result = select_manuscript_sparse_support(
            input_matrix,
            feature_catalog,
            holdout_assignments,
            pca_scores,
            retained_terms,
            retained_pairs,
            retained_transformations,
            sparse_spec,
        )
        stable = sparse_result.final_stable_support
        if not stable.empty:
            final_selected_support = frozenset(stable["feature_name"].astype(str))
    except ValueError:
        pass

    # ------------------------------------------------------------------
    # 9. Final-OLS predictions on X_eval
    # ------------------------------------------------------------------
    eval_predictions = _ols_eval_predictions(
        input_matrix=input_matrix,
        Y_train=Y_tr_arr,
        final_support=final_selected_support,
        train_ids=train_ids,
        eval_ids=eval_ids,
        n_eval=n_eval,
        n_outputs=n_outputs,
    )

    return ProductionRecoveryResult(
        screening_candidate_count=screening_candidate_count,
        screening_retained_set=screening_retained_set,
        interaction_candidate_count=interaction_candidate_count,
        interaction_retained_set=interaction_retained_set,
        nonlinear_candidate_count=nonlinear_candidate_count,
        nonlinear_retained_set=nonlinear_retained_set,
        final_selected_support=final_selected_support,
        eval_predictions=eval_predictions,
    )


def _ols_eval_predictions(
    *,
    input_matrix: pd.DataFrame,
    Y_train: np.ndarray,
    final_support: frozenset[str],
    train_ids: list[int],
    eval_ids: list[int],
    n_eval: int,
    n_outputs: int,
) -> np.ndarray:
    """Fit OLS on (X_train_design, Y_train) and return predictions on X_eval_design.

    Falls back to column-mean null predictions when the final support is empty
    or any design-building step fails.
    """
    null_pred = np.tile(Y_train.mean(axis=0), (n_eval, 1))
    if not final_support:
        return null_pred

    support_list = sorted(final_support)
    support_catalog = pd.DataFrame(
        {
            "feature_name": support_list,
            "feature_type": ["first_order"] * len(support_list),
        }
    )
    try:
        train_rows = input_matrix[input_matrix["sample_id"].isin(train_ids)].reset_index(drop=True)
        eval_rows = input_matrix[input_matrix["sample_id"].isin(eval_ids)].reset_index(drop=True)

        X_design_tr = build_manuscript_feature_design(train_rows, support_catalog)
        X_design_ev = build_manuscript_feature_design(eval_rows, support_catalog)

        X_tr = X_design_tr.drop(columns=["sample_id"]).to_numpy(dtype=float)
        X_ev = X_design_ev.drop(columns=["sample_id"]).to_numpy(dtype=float)

        # Add intercept column
        ones_tr = np.ones((len(X_tr), 1), dtype=float)
        ones_ev = np.ones((len(X_ev), 1), dtype=float)
        X_tr = np.hstack([ones_tr, X_tr])
        X_ev = np.hstack([ones_ev, X_ev])

        beta, _, _, _ = np.linalg.lstsq(X_tr, Y_train, rcond=None)
        return X_ev @ beta
    except Exception:  # noqa: BLE001
        return null_pred


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ProductionRecoveryResult",
    "RecoveryScenario",
    "empirical_interaction_fwer",
    "prespecified_recovery_scenarios",
    "recovery_estimands",
    "run_production_recovery_pipeline",
]
