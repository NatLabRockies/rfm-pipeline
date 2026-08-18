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

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from .campaign_contract import (
    G11_CONTRACT,
    CampaignContract,
    compute_contract_hash,
    validate_contract_hash,
)
from .interaction_contract import (
    ScoreOnlyInteractionArtifact,
    canonical_execution_contract_from_specs,
)
from .manuscript_stages import (
    EmpiricalNullScreeningSpec,
    InteractionDiscoverySpec,
    NonlinearDiscoverySpec,
    OutputConditioningSpec,
    SparseSelectionStabilitySpec,
    build_manuscript_feature_design,
    condition_manuscript_outputs,
    discover_manuscript_nonlinear_transformations,
    holdout_predict,
    reduce_score_only_interaction_artifacts,
    score_interaction_draw_block,
    screen_manuscript_empirical_null_terms,
    select_manuscript_sparse_support,
    terminal_train_fit_and_freeze,
    train_fit_and_freeze,
    write_frozen_prediction_matrices,
    write_terminal_train_fit_and_freeze,
    write_train_fit_and_freeze,
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
    terminal_status: str
    contract_hash: str
    interaction_artifact_checksums: tuple[str, ...]
    model_freeze_hash: str | None
    algebraic_candidate_names: tuple[str, ...] = ()
    comparator_predictions: dict[str, np.ndarray] | None = None
    comparator_schedule_sha256: str | None = None
    comparator_hyperparameters: dict[str, object] | None = None


def run_production_recovery_pipeline(
    X_train: np.ndarray | pd.DataFrame,
    Y_train: np.ndarray | pd.DataFrame,
    X_eval: np.ndarray | pd.DataFrame,
    Y_eval: np.ndarray | pd.DataFrame,
    *,
    execution_contract: CampaignContract = G11_CONTRACT,
    artifact_dir: Path | None = None,
    seed: int = 0,
) -> ProductionRecoveryResult:
    """Run the frozen G11 path through persisted interaction shards and terminal fit.

    The sole scientific-settings input is ``CampaignContract``. Screening and
    interaction use separate permutation counts, interaction decisions are made
    only after byte round-tripping at least two score artifacts for a non-empty
    family, and evaluation responses are not converted or opened until HC3,
    pruning, final OLS, and the model freeze have completed.
    """
    contract_hash = compute_contract_hash(execution_contract)
    validate_contract_hash(execution_contract, contract_hash)
    if execution_contract.method_name != "max_stat_adjusted_p_mc":
        raise ValueError("recovery execution requires method_name=max_stat_adjusted_p_mc")
    if execution_contract.B_screen < 3199 or execution_contract.B_interaction < 999:
        raise ValueError("recovery execution contract violates the G11 permutation floors")
    if execution_contract.n_tree_estimators != 250:
        raise ValueError("recovery execution contract requires exactly 250 tree estimators")
    if execution_contract.n_stability_subsamples != 50:
        raise ValueError("recovery execution contract requires exactly 50 stability subsamples")
    if artifact_dir is None:
        with TemporaryDirectory(prefix="rfm-recovery-") as temporary:
            return run_production_recovery_pipeline(
                X_train,
                Y_train,
                X_eval,
                Y_eval,
                execution_contract=execution_contract,
                artifact_dir=Path(temporary),
                seed=seed,
            )
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=False)

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

    n_train, n_inputs = X_tr_arr.shape
    n_outputs = Y_tr_arr.shape[1]
    if X_tr_arr.shape[0] != Y_tr_arr.shape[0]:
        raise ValueError("training input and response rows must match")
    expected_inputs = {
        scenario.n_predictor_continuous + scenario.n_predictor_binary
        for scenario in execution_contract.scenarios
    }
    if expected_inputs != {160} or n_inputs != 160:
        raise ValueError("G11 recovery execution requires 158 continuous plus 2 binary inputs")

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

    # ------------------------------------------------------------------
    # 2. Build combined DataFrames for the stage functions
    # ------------------------------------------------------------------
    input_matrix = pd.DataFrame(
        {"sample_id": train_ids, **{name: X_tr_arr[:, i] for i, name in enumerate(input_names)}}
    )
    output_matrix = pd.DataFrame(
        {"sample_id": train_ids, **{name: Y_tr_arr[:, j] for j, name in enumerate(output_names)}}
    )

    holdout_assignments = pd.DataFrame(
        {
            "sample_id": train_ids,
            "split": ["train"] * n_train,
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
        retained_components=min(2, n_outputs),
        retained_variance_fraction=0.99,
    )
    screening_spec = EmpiricalNullScreeningSpec(
        statistic="coefficient_row_l2_norm",
        permutation_count_B=execution_contract.B_screen,
        bh_q_screen=execution_contract.q_screen,
        retained_terms_reference=0,
        random_seed=seed,
    )
    interaction_spec = InteractionDiscoverySpec(
        method=execution_contract.interaction_detector_method,
        aggregation_rule="max_over_components_by_detector",
        null_threshold_quantile=0.95,
        retained_pairs_reference=0,
        permutation_count_B=execution_contract.B_interaction,
        random_seed=seed,
        selection_method=execution_contract.method_name,
        selection_alpha=execution_contract.alpha,
        tree_family_alpha=execution_contract.tree_family_alpha,
        binary_binary_family_alpha=execution_contract.binary_binary_family_alpha,
        family_partition_method=execution_contract.family_partition_method,
        binary_binary_method=execution_contract.binary_binary_method,
        binary_binary_minimum_cell_count=(execution_contract.binary_binary_minimum_cell_count),
        enforce_permutation_adequacy=True,
        minimum_selection_draws=execution_contract.B_interaction,
        n_jobs=1,
        n_tree_estimators=execution_contract.n_tree_estimators,
        max_shap_samples=n_train,
    )
    nonlinear_spec = NonlinearDiscoverySpec(
        method="gam_plus_restricted_parametric_replacement",
        curvature_rule="edf_gt_1_and_smooth_pvalue_lt_0p01",
        replacement_selection_rule="minimum_training_rmse_against_gam_smooth",
        identified_transformations_reference=0,
        final_support_transformations_reference=0,
        family_wise_alpha=execution_contract.alpha,
        cv_seed=seed,
    )
    sparse_spec = SparseSelectionStabilitySpec(
        model_class="l1_penalized_linear_model_per_retained_component",
        ebic_gamma=0.5,
        subsample_count=execution_contract.n_stability_subsamples,
        subsample_fraction=0.8,
        jaccard_threshold=execution_contract.stability_jaccard_threshold,
        spearman_threshold=execution_contract.stability_spearman_threshold,
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

    # Interaction scoring must cross the exact persistence boundary used by HPC.
    canonical_contract = canonical_execution_contract_from_specs(interaction_spec)
    interaction_root = artifact_dir / "interaction_score_blocks"
    interaction_root.mkdir()
    if len(retained_terms) < 2:
        draw_ranges = [(0, execution_contract.B_interaction)]
    else:
        split = execution_contract.B_interaction // 2
        draw_ranges = [(0, split), (split, execution_contract.B_interaction)]
    persisted_interactions: list[ScoreOnlyInteractionArtifact] = []
    for block_index, (draw_start, draw_end) in enumerate(draw_ranges):
        artifact = score_interaction_draw_block(
            input_matrix,
            feature_catalog,
            holdout_assignments,
            pca_scores,
            retained_terms,
            interaction_spec,
            draw_start=draw_start,
            draw_end=draw_end,
            contract=canonical_contract,
        )
        block_path = interaction_root / f"block-{block_index:02d}"
        artifact.write_to(block_path)
        persisted_interactions.append(ScoreOnlyInteractionArtifact.read_from(block_path))
    ix_result = reduce_score_only_interaction_artifacts(
        persisted_interactions,
        spec=interaction_spec,
        contract=canonical_contract,
    )
    interaction_checksums = tuple(artifact.payload_sha256 for artifact in persisted_interactions)
    retained_pairs = ix_result.retained_pairs
    interaction_candidate_count = len(ix_result.pair_scores)
    interaction_retained_set = frozenset(retained_pairs["pair_name"].astype(str))

    if not screening_retained_set:
        intercept_freeze = train_fit_and_freeze(
            np.zeros((n_train, 1), dtype=float),
            Y_tr_arr,
            feature_names=["__intercept_only__"],
            output_names=output_names,
            contract_hash=contract_hash,
        )
        intercept_root = artifact_dir / "terminal_train_fit" / "model"
        write_train_fit_and_freeze(
            freeze_result=intercept_freeze,
            output_dir=intercept_root,
        )
        X_ev_arr = _to_arr(X_eval)
        if X_ev_arr.shape[1] != n_inputs:
            raise ValueError("training and evaluation inputs must have the same columns")
        n_eval = X_ev_arr.shape[0]
        eval_ids = list(range(n_train, n_train + n_eval))
        y_eval_arr = _to_arr(Y_eval)
        if y_eval_arr.shape != (n_eval, n_outputs):
            raise ValueError("evaluation response shape must match evaluation rows and outputs")
        frozen_predictions = holdout_predict(
            intercept_freeze,
            np.zeros((n_eval, 1), dtype=float),
            y_eval_arr,
            holdout_ids=tuple(f"eval-{index:06d}" for index in range(n_eval)),
            strata=np.asarray(["recovery_eval"] * n_eval),
        )
        write_frozen_prediction_matrices(
            frozen=frozen_predictions,
            output_dir=artifact_dir / "frozen_eval_predictions",
        )
        return ProductionRecoveryResult(
            screening_candidate_count=screening_candidate_count,
            screening_retained_set=screening_retained_set,
            interaction_candidate_count=interaction_candidate_count,
            interaction_retained_set=interaction_retained_set,
            nonlinear_candidate_count=0,
            nonlinear_retained_set=frozenset(),
            final_selected_support=frozenset(),
            eval_predictions=frozen_predictions.y_pred,
            terminal_status="empty_candidate_family",
            contract_hash=contract_hash,
            interaction_artifact_checksums=interaction_checksums,
            model_freeze_hash=intercept_freeze.freeze_manifest.freeze_hash,
            algebraic_candidate_names=(),
        )

    # ------------------------------------------------------------------
    # 7. Stage 4: nonlinear discovery
    # ------------------------------------------------------------------
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
    nonlinear_retained_set = frozenset(retained_transformations["feature_name"].astype(str))
    algebraic_candidate_names = tuple(
        dict.fromkeys(
            [
                *retained_terms["feature_name"].astype(str).tolist(),
                *retained_pairs["pair_name"].astype(str).tolist(),
                *retained_transformations["feature_name"].astype(str).tolist(),
            ]
        )
    )

    # ------------------------------------------------------------------
    # 8. Stage 5: sparse EBIC selection + stability filtering
    # ------------------------------------------------------------------
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
    stable_names = list(sparse_result.final_stable_support["feature_name"].astype(str))
    if not stable_names:
        raise ValueError("sparse/stability selection retained no terminal candidates")
    support_catalog = pd.DataFrame({"feature_name": stable_names})
    x_train_design = build_manuscript_feature_design(input_matrix, support_catalog)
    x_train_frame = x_train_design.drop(columns=["sample_id"])
    x_train_frame.index = pd.Index(train_ids, name="sample_id")
    y_train_frame = pd.DataFrame(Y_tr_arr, index=x_train_frame.index, columns=output_names)

    from .manuscript_stages import FinalManuscriptArtifactsSpec

    final_spec = FinalManuscriptArtifactsSpec(
        final_predictor_count_reference=0,
        final_first_order_input_count_reference=0,
        intermediate_penalized_holdout_nrmse_reference=0.0,
        final_ols_holdout_nrmse_reference=0.0,
        nrmse_denominator_definition="training_response_range",
        nrmse_min_range=1.0e-12,
        nrmse_reference_matrix="Y_train",
        bootstrap_count=2,
        random_seed=seed,
        inferential_filter_interval_method=(
            "hc3_wald_95_percent_drop_if_zero_compatible_for_all_outputs"
        ),
        inferential_filter_alpha=0.05,
        hc3_output_subset_mode="all",
        pruning_error_scale_quantile=0.95,
        n_jobs=1,
    )
    terminal = terminal_train_fit_and_freeze(
        x_train_frame,
        y_train_frame,
        spec=final_spec,
        contract_hash=contract_hash,
    )
    terminal_root = artifact_dir / "terminal_train_fit"
    write_terminal_train_fit_and_freeze(terminal=terminal, output_dir=terminal_root)
    final_selected_support = frozenset(terminal.final_feature_names)

    # Only after the terminal model and manifest are frozen do we materialize
    # evaluation responses or authorize prediction.
    X_ev_arr = _to_arr(X_eval)
    if X_ev_arr.shape[1] != n_inputs:
        raise ValueError("training and evaluation inputs must have the same columns")
    n_eval = X_ev_arr.shape[0]
    eval_ids = list(range(n_train, n_train + n_eval))
    eval_input_matrix = pd.DataFrame(
        {"sample_id": eval_ids, **{name: X_ev_arr[:, i] for i, name in enumerate(input_names)}}
    )
    x_eval_design = build_manuscript_feature_design(eval_input_matrix, support_catalog)
    x_eval_final = x_eval_design.loc[:, list(terminal.final_feature_names)].to_numpy(dtype=float)
    y_eval_arr = _to_arr(Y_eval)
    if y_eval_arr.shape != (n_eval, n_outputs):
        raise ValueError("evaluation response shape must match evaluation rows and outputs")
    frozen_predictions = holdout_predict(
        terminal.freeze_result,
        x_eval_final,
        y_eval_arr,
        holdout_ids=tuple(f"eval-{index:06d}" for index in range(n_eval)),
        strata=np.asarray(["recovery_eval"] * n_eval),
    )
    write_frozen_prediction_matrices(
        frozen=frozen_predictions,
        output_dir=artifact_dir / "frozen_eval_predictions",
    )
    eval_predictions = frozen_predictions.y_pred

    return ProductionRecoveryResult(
        screening_candidate_count=screening_candidate_count,
        screening_retained_set=screening_retained_set,
        interaction_candidate_count=interaction_candidate_count,
        interaction_retained_set=interaction_retained_set,
        nonlinear_candidate_count=nonlinear_candidate_count,
        nonlinear_retained_set=nonlinear_retained_set,
        final_selected_support=final_selected_support,
        eval_predictions=eval_predictions,
        terminal_status="completed",
        contract_hash=contract_hash,
        interaction_artifact_checksums=interaction_checksums,
        model_freeze_hash=terminal.freeze_result.freeze_manifest.freeze_hash,
        algebraic_candidate_names=algebraic_candidate_names,
    )


@dataclass(frozen=True)
class RecoveryComparatorResult:
    """Training-only tuned prediction matrices for the four frozen comparators."""

    predictions: dict[str, np.ndarray]
    selected_hyperparameters: dict[str, object]
    schedule_sha256: str
    contract_hash: str


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _ols_predictions(
    train_design: np.ndarray, y_train: np.ndarray, eval_design: np.ndarray
) -> np.ndarray:
    train_augmented = np.column_stack([np.ones(train_design.shape[0]), train_design])
    eval_augmented = np.column_stack([np.ones(eval_design.shape[0]), eval_design])
    coefficients, _, _, _ = np.linalg.lstsq(train_augmented, y_train, rcond=None)
    return np.asarray(eval_augmented @ coefficients, dtype=float)


def run_recovery_comparators(
    *,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    oracle_train_design: np.ndarray,
    oracle_eval_design: np.ndarray,
    algebraic_train_design: np.ndarray,
    algebraic_eval_design: np.ndarray,
    proposed_predictions: np.ndarray,
    execution_contract: CampaignContract = G11_CONTRACT,
    seed: int,
) -> RecoveryComparatorResult:
    """Fit frozen comparators without accepting or reading evaluation responses."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import MultiTaskElasticNetCV
    from sklearn.model_selection import KFold
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.preprocessing import StandardScaler

    expected_names = (
        "proposed_terminal_workflow",
        "oracle_ols",
        "elastic_net_algebraic_library",
        "raw_input_boosted_tree",
    )
    if execution_contract.recovery_comparators != expected_names:
        raise ValueError("recovery comparator names differ from the frozen contract")
    arrays = {
        "X_train": np.asarray(X_train, dtype=float),
        "Y_train": np.asarray(Y_train, dtype=float),
        "X_eval": np.asarray(X_eval, dtype=float),
        "oracle_train": np.asarray(oracle_train_design, dtype=float),
        "oracle_eval": np.asarray(oracle_eval_design, dtype=float),
        "algebraic_train": np.asarray(algebraic_train_design, dtype=float),
        "algebraic_eval": np.asarray(algebraic_eval_design, dtype=float),
        "proposed": np.asarray(proposed_predictions, dtype=float),
    }
    if any(array.ndim != 2 or not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("recovery comparator inputs must be finite two-dimensional arrays")
    n_train, n_outputs = arrays["Y_train"].shape
    n_eval = arrays["X_eval"].shape[0]
    if arrays["X_train"].shape[0] != n_train:
        raise ValueError("raw training inputs and responses have different rows")
    if arrays["X_train"].shape[1] != arrays["X_eval"].shape[1]:
        raise ValueError("raw evaluation design differs from raw training columns")
    if arrays["oracle_train"].shape[0] != n_train:
        raise ValueError("oracle training design differs from training rows")
    if arrays["oracle_eval"].shape != (n_eval, arrays["oracle_train"].shape[1]):
        raise ValueError("oracle evaluation design differs from the frozen oracle columns")
    if arrays["algebraic_train"].shape[0] != n_train:
        raise ValueError("algebraic training design differs from training rows")
    if arrays["algebraic_eval"].shape != (
        n_eval,
        arrays["algebraic_train"].shape[1],
    ):
        raise ValueError("algebraic evaluation design differs from the frozen candidate library")
    if arrays["proposed"].shape != (n_eval, n_outputs):
        raise ValueError("proposed predictions differ from the frozen evaluation dimensions")

    cv = KFold(
        n_splits=execution_contract.comparator_cv_folds,
        shuffle=True,
        random_state=seed,
    )
    algebraic_scaler = StandardScaler()
    algebraic_train_scaled = algebraic_scaler.fit_transform(arrays["algebraic_train"])
    algebraic_eval_scaled = algebraic_scaler.transform(arrays["algebraic_eval"])
    elastic = MultiTaskElasticNetCV(
        l1_ratio=list(execution_contract.elastic_net_l1_ratio_grid),
        alphas=np.asarray(execution_contract.elastic_net_alpha_grid, dtype=float),
        cv=cv,
        max_iter=5000,
        n_jobs=1,
        selection="cyclic",
    )
    elastic.fit(algebraic_train_scaled, arrays["Y_train"])
    elastic_predictions = np.asarray(elastic.predict(algebraic_eval_scaled), dtype=float)

    response_components = min(
        execution_contract.boosted_tree_response_components,
        n_outputs,
        n_train - 1,
    )
    if response_components < 1:
        raise ValueError("boosted-tree comparator requires an estimable response component")
    candidate_scores: list[tuple[float, tuple[int, int, float]]] = []
    for candidate_index, candidate in enumerate(execution_contract.boosted_tree_candidate_grid):
        estimators, depth, learning_rate = candidate
        fold_scores = []
        for fold_index, (fit_rows, score_rows) in enumerate(cv.split(arrays["X_train"])):
            y_fit = arrays["Y_train"][fit_rows]
            y_mean = y_fit.mean(axis=0)
            _, _, right_vectors = np.linalg.svd(y_fit - y_mean, full_matrices=False)
            basis = right_vectors[:response_components]
            fit_scores = (y_fit - y_mean) @ basis.T
            model = MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=estimators,
                    max_depth=depth,
                    learning_rate=learning_rate,
                    random_state=seed + candidate_index * 100 + fold_index,
                ),
                n_jobs=1,
            )
            model.fit(arrays["X_train"][fit_rows], fit_scores)
            predicted_scores = np.asarray(
                model.predict(arrays["X_train"][score_rows]), dtype=float
            ).reshape(len(score_rows), response_components)
            prediction = predicted_scores @ basis + y_mean
            output_ranges = np.ptp(y_fit, axis=0)
            eligible = output_ranges > 1.0e-12
            if not eligible.any():
                raise ValueError("boosted-tree tuning fold has no variable responses")
            per_output_rmse = np.sqrt(
                np.mean(
                    (arrays["Y_train"][score_rows] - prediction) ** 2,
                    axis=0,
                )
            )
            fold_scores.append(float(np.mean(per_output_rmse[eligible] / output_ranges[eligible])))
        candidate_scores.append((float(np.mean(fold_scores)), candidate))
    boosted_score, boosted_candidate = min(candidate_scores, key=lambda value: (value[0], value[1]))
    estimators, depth, learning_rate = boosted_candidate
    response_mean = arrays["Y_train"].mean(axis=0)
    _, _, right_vectors = np.linalg.svd(
        arrays["Y_train"] - response_mean,
        full_matrices=False,
    )
    response_basis = right_vectors[:response_components]
    response_scores = (arrays["Y_train"] - response_mean) @ response_basis.T
    boosted = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=estimators,
            max_depth=depth,
            learning_rate=learning_rate,
            random_state=seed,
        ),
        n_jobs=1,
    )
    boosted.fit(arrays["X_train"], response_scores)
    boosted_eval_scores = np.asarray(boosted.predict(arrays["X_eval"]), dtype=float).reshape(
        n_eval, response_components
    )
    boosted_predictions = boosted_eval_scores @ response_basis + response_mean

    predictions = {
        "proposed_terminal_workflow": arrays["proposed"].copy(),
        "oracle_ols": _ols_predictions(
            arrays["oracle_train"], arrays["Y_train"], arrays["oracle_eval"]
        ),
        "elastic_net_algebraic_library": elastic_predictions,
        "raw_input_boosted_tree": boosted_predictions,
    }
    if any(
        value.shape != (n_eval, n_outputs) or not np.isfinite(value).all()
        for value in predictions.values()
    ):
        raise ValueError("a frozen recovery comparator emitted invalid predictions")
    selected = {
        "elastic_net": {
            "alpha": float(elastic.alpha_),
            "l1_ratio": float(elastic.l1_ratio_),
        },
        "boosted_tree": {
            "n_estimators": estimators,
            "max_depth": depth,
            "learning_rate": learning_rate,
            "training_cv_score": boosted_score,
            "response_components": response_components,
        },
    }
    schedule = {
        "seed": seed,
        "cv_folds": execution_contract.comparator_cv_folds,
        "elastic_net_alpha_grid": execution_contract.elastic_net_alpha_grid,
        "elastic_net_l1_ratio_grid": execution_contract.elastic_net_l1_ratio_grid,
        "boosted_tree_candidate_grid": execution_contract.boosted_tree_candidate_grid,
        "boosted_tree_response_components": response_components,
        "tuning_metric": execution_contract.comparator_tuning_metric,
    }
    return RecoveryComparatorResult(
        predictions=predictions,
        selected_hyperparameters=selected,
        schedule_sha256=_canonical_sha256(schedule),
        contract_hash=compute_contract_hash(execution_contract),
    )


def write_recovery_comparator_result(
    result: RecoveryComparatorResult, output_dir: Path
) -> dict[str, Path]:
    """Persist comparator matrices and their exact training-only tuning identity."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    predictions_path = root / "comparator_predictions.npz"
    np.savez_compressed(predictions_path, **result.predictions)
    predictions_sha256 = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "contract_hash": result.contract_hash,
        "schedule_sha256": result.schedule_sha256,
        "selected_hyperparameters": result.selected_hyperparameters,
        "prediction_names": list(result.predictions),
        "predictions_sha256": predictions_sha256,
    }
    metadata_path = root / "comparator_predictions.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"predictions": predictions_path, "metadata": metadata_path}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ProductionRecoveryResult",
    "RecoveryComparatorResult",
    "RecoveryScenario",
    "empirical_interaction_fwer",
    "prespecified_recovery_scenarios",
    "recovery_estimands",
    "run_recovery_comparators",
    "run_production_recovery_pipeline",
    "write_recovery_comparator_result",
]
