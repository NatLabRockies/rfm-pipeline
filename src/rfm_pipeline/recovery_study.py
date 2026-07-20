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
from scipy.stats import norm as _norm

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
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "RecoveryScenario",
    "empirical_interaction_fwer",
    "prespecified_recovery_scenarios",
    "recovery_estimands",
]
