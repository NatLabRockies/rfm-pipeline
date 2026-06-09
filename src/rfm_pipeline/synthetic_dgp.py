# Copyright (c) 2026 Dylan Hettinger
"""Standalone synthetic data-generating processes for sensitivity studies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from math import exp, pi

import numpy as np
import pandas as pd
from scipy.special import ndtr

_TRUE_NONLINEAR_TRANSFORMS = ("sin", "sq", "exp")
_CATALOG_NONLINEAR_TRANSFORMS = ("sin", "sq", "log1p", "exp")


@dataclass(frozen=True, slots=True)
class SyntheticDGPSpec:
    """Parameterized synthetic DGP definition."""

    n_inputs: int
    n_runs: int
    n_outputs: int
    sparsity: float
    interaction_density: float
    nonlinearity_strength: float
    noise_snr: float
    holdout_fraction: float
    dgp_family: str
    seed: int
    factor_model_rank: int = 20
    input_correlation_strength: float = 0.3
    factor_signal_weight: float = 1.0
    output_scale_heterogeneity: float = 0.0
    output_nonlinearity_strength: float = 0.0
    per_output_snr_heterogeneity: float = 0.0
    active_input_beta_concentration: float = 0.0

    def __post_init__(self) -> None:
        """Validate all spec fields at construction time."""
        _validate_positive_int(self.n_inputs, "n_inputs")
        _validate_positive_int(self.n_runs, "n_runs")
        _validate_positive_int(self.n_outputs, "n_outputs")
        _validate_range(self.sparsity, "sparsity", 0.05, 0.70)
        _validate_range(self.interaction_density, "interaction_density", 0.0, 0.5)
        _validate_range(self.nonlinearity_strength, "nonlinearity_strength", 0.0, 1.0)
        _validate_range(self.noise_snr, "noise_snr", 5.0, 100.0)
        _validate_range(self.holdout_fraction, "holdout_fraction", 0.05, 0.20)
        if self.dgp_family not in {"pure_synthetic", "bsm_structure", "calibrated_structure"}:
            raise ValueError(
                f"dgp_family must be 'pure_synthetic', 'bsm_structure', or"
                f" 'calibrated_structure'; got {self.dgp_family!r}"
            )
        _validate_positive_int(self.factor_model_rank, "factor_model_rank")
        _validate_range(
            self.input_correlation_strength,
            "input_correlation_strength",
            0.0,
            0.95,
        )
        _validate_range(self.factor_signal_weight, "factor_signal_weight", 0.0, 100.0)
        _validate_range(
            self.output_scale_heterogeneity,
            "output_scale_heterogeneity",
            0.0,
            3.0,
        )
        _validate_range(
            self.output_nonlinearity_strength,
            "output_nonlinearity_strength",
            0.0,
            2.0,
        )
        _validate_range(
            self.per_output_snr_heterogeneity,
            "per_output_snr_heterogeneity",
            0.0,
            3.0,
        )
        _validate_range(
            self.active_input_beta_concentration,
            "active_input_beta_concentration",
            0.0,
            3.0,
        )


@dataclass(frozen=True, slots=True)
class DGPTrueSupport:
    """Truth set used for support-recovery evaluation."""

    true_active_inputs: frozenset[str]
    true_active_interactions: frozenset[tuple[str, str]]
    true_active_nonlinear: frozenset[str]


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """Synthetic tables matching manuscript runtime contracts."""

    input_matrix: pd.DataFrame
    output_matrix: pd.DataFrame
    holdout_assignments: pd.DataFrame
    feature_catalog: pd.DataFrame
    true_support: DGPTrueSupport
    spec: SyntheticDGPSpec


@dataclass(frozen=True, slots=True)
class _SignalBundle:
    signals: np.ndarray
    active_inputs: tuple[int, ...]
    active_interactions: frozenset[tuple[str, str]]
    active_nonlinear: frozenset[str]


def generate_pure_synthetic(spec: SyntheticDGPSpec) -> SyntheticDataset:
    """Generate independent-input synthetic data."""
    if spec.dgp_family != "pure_synthetic":
        raise ValueError(
            "generate_pure_synthetic requires spec.dgp_family='pure_synthetic'; "
            f"got {spec.dgp_family!r}"
        )
    rng = np.random.default_rng(spec.seed)
    inputs = rng.random((spec.n_runs, spec.n_inputs), dtype=np.float64)
    return _assemble_dataset(spec=spec, inputs=inputs, rng=rng, factor_signal=None)


def generate_calibrated_structure_synthetic(spec: SyntheticDGPSpec) -> SyntheticDataset:
    """Generate correlated-input/factor-output synthetic data."""
    if spec.dgp_family not in {"bsm_structure", "calibrated_structure"}:
        raise ValueError(
            "generate_calibrated_structure_synthetic requires dgp_family"
            " 'bsm_structure' or 'calibrated_structure'; "
            f"got {spec.dgp_family!r}"
        )
    rng = np.random.default_rng(spec.seed)
    corr = _ar1_correlation_matrix(spec.n_inputs, spec.input_correlation_strength)
    chol = np.linalg.cholesky(corr)
    gaussian = rng.normal(size=(spec.n_runs, spec.n_inputs)) @ chol.T
    inputs = np.clip(ndtr(gaussian), 1e-9, 1.0 - 1e-9)

    latent = rng.normal(size=(spec.n_runs, spec.factor_model_rank))
    loadings = rng.normal(size=(spec.factor_model_rank, spec.n_outputs))
    factor_signal = latent @ loadings
    factor_scale = factor_signal.std(axis=0, ddof=1)
    factor_scale[factor_scale == 0.0] = 1.0
    factor_signal = factor_signal / factor_scale
    if spec.factor_signal_weight != 1.0:
        factor_signal = factor_signal * spec.factor_signal_weight
    return _assemble_dataset(spec=spec, inputs=inputs, rng=rng, factor_signal=factor_signal)


def _assemble_dataset(
    *,
    spec: SyntheticDGPSpec,
    inputs: np.ndarray,
    rng: np.random.Generator,
    factor_signal: np.ndarray | None,
) -> SyntheticDataset:
    signal_bundle = _generate_sparse_signals(spec=spec, inputs=inputs, rng=rng)
    total_signal = signal_bundle.signals.copy()
    if factor_signal is not None:
        total_signal += factor_signal

    signal_var = total_signal.var(axis=0, ddof=1)
    signal_var = np.maximum(signal_var, 1e-12)

    # Optional per-output SNR heterogeneity: noise floor varies output-to-output. Drives the
    # std/range RATIO (which is otherwise scale-invariant) and creates per-output heterogeneity
    # that the scalar SNR cannot reproduce. Real datasets typically have outputs with very
    # different noise characteristics.
    if spec.per_output_snr_heterogeneity > 0.0:
        log_snr_jitter = rng.normal(
            loc=0.0, scale=spec.per_output_snr_heterogeneity, size=spec.n_outputs
        )
        per_output_snr = spec.noise_snr * np.exp(log_snr_jitter)
        noise_std = np.sqrt(signal_var / per_output_snr)
    else:
        noise_std = np.sqrt(signal_var / spec.noise_snr)
    outputs = total_signal + rng.normal(scale=noise_std, size=total_signal.shape)

    # Optional output-scale heterogeneity: per-output multiplicative scale ~ LogNormal(0, sigma).
    # Drives PCA top-1 share (a few large-scale outputs dominate the centered SVD spectrum).
    if spec.output_scale_heterogeneity > 0.0:
        scale_factors = np.exp(
            rng.normal(loc=0.0, scale=spec.output_scale_heterogeneity, size=spec.n_outputs)
        )
        outputs = outputs * scale_factors

    # Optional monotonic asymmetric post-nonlinearity. Per-output strength sampled from
    # Uniform(0, s_max) so MOST outputs are near-identity while a SUBSET is strongly transformed.
    # This produces datasets where skewness can be high in aggregate while median kurtosis stays
    # moderate -- a regime that covers many real-world bounded/log-like output families.
    if spec.output_nonlinearity_strength > 0.0:
        s_per_output = rng.uniform(0.0, spec.output_nonlinearity_strength, size=spec.n_outputs)
        scale = outputs.std(axis=0, ddof=1)
        scale = np.where(scale > 1e-12, scale, 1.0)
        active = s_per_output > 0.01
        if np.any(active):
            s_row = np.where(active, s_per_output, 1.0)[None, :]
            transformed = np.expm1(s_row * outputs / scale[None, :]) / s_row
            outputs = np.where(active[None, :], transformed, outputs)

    sample_ids = np.array([f"s{i:06d}" for i in range(spec.n_runs)], dtype=object)
    input_columns = [f"x{i}" for i in range(spec.n_inputs)]
    output_columns = [f"y{i}" for i in range(spec.n_outputs)]

    input_matrix = pd.DataFrame(inputs, columns=input_columns)
    input_matrix.insert(0, "sample_id", sample_ids)

    output_matrix = pd.DataFrame(outputs, columns=output_columns)
    output_matrix.insert(0, "sample_id", sample_ids)

    holdout_assignments = _make_holdout_assignments(sample_ids=sample_ids, spec=spec)
    feature_catalog = _build_feature_catalog(
        spec=spec,
        true_interactions=signal_bundle.active_interactions,
    )

    support = DGPTrueSupport(
        true_active_inputs=frozenset(input_columns[index] for index in signal_bundle.active_inputs),
        true_active_interactions=signal_bundle.active_interactions,
        true_active_nonlinear=signal_bundle.active_nonlinear,
    )
    return SyntheticDataset(
        input_matrix=input_matrix,
        output_matrix=output_matrix,
        holdout_assignments=holdout_assignments,
        feature_catalog=feature_catalog,
        true_support=support,
        spec=spec,
    )


def _generate_sparse_signals(
    *,
    spec: SyntheticDGPSpec,
    inputs: np.ndarray,
    rng: np.random.Generator,
) -> _SignalBundle:
    centered = inputs - 0.5
    n_active = max(1, round(spec.sparsity * spec.n_inputs))
    active_inputs = tuple(sorted(rng.choice(spec.n_inputs, size=n_active, replace=False).tolist()))
    signals = np.zeros((spec.n_runs, spec.n_outputs), dtype=np.float64)
    active_interactions: set[tuple[str, str]] = set()
    active_nonlinear: set[str] = set()

    interaction_pairs = list(combinations(active_inputs, 2))
    n_active = len(active_inputs)
    beta_alpha = spec.active_input_beta_concentration
    # Per-active-input geometric weighting. At alpha=0 (default), weights are uniform 1.0
    # (current behavior). At alpha>0, first active input gets dominant beta, rest fall off as
    # i^(-alpha) -- produces Pareto-like signal concentration where a few inputs dominate
    # marginal X_j-Y correlations.
    if beta_alpha > 0.0:
        beta_weights = np.power(np.arange(1, n_active + 1, dtype=np.float64), -beta_alpha)
    else:
        beta_weights = np.ones(n_active, dtype=np.float64)

    for output_idx in range(spec.n_outputs):
        signal = np.zeros(spec.n_runs, dtype=np.float64)
        betas = rng.normal(size=n_active) * beta_weights
        signal += centered[:, active_inputs] @ betas

        for left_idx, right_idx in interaction_pairs:
            if rng.random() < spec.interaction_density:
                gamma = rng.normal()
                signal += gamma * centered[:, left_idx] * centered[:, right_idx]
                active_interactions.add(_ordered_feature_pair(left_idx, right_idx))

        if spec.nonlinearity_strength > 0.0:
            for input_idx in active_inputs:
                transform_name = _TRUE_NONLINEAR_TRANSFORMS[
                    int(rng.integers(0, len(_TRUE_NONLINEAR_TRANSFORMS)))
                ]
                delta = rng.normal()
                signal += (
                    spec.nonlinearity_strength
                    * delta
                    * _evaluate_transform(transform_name, inputs[:, input_idx])
                )
                active_nonlinear.add(f"{transform_name}_x{input_idx}")

        signals[:, output_idx] = signal

    return _SignalBundle(
        signals=signals,
        active_inputs=active_inputs,
        active_interactions=frozenset(sorted(active_interactions, key=_pair_sort_key)),
        active_nonlinear=frozenset(sorted(active_nonlinear, key=_feature_name_sort_key)),
    )


def _make_holdout_assignments(*, sample_ids: np.ndarray, spec: SyntheticDGPSpec) -> pd.DataFrame:
    holdout_count = round(spec.n_runs * spec.holdout_fraction)
    if spec.n_runs > 1:
        holdout_count = min(spec.n_runs - 1, max(1, holdout_count))
    else:
        holdout_count = 0
    split = np.full(spec.n_runs, "train", dtype=object)
    if holdout_count:
        split[-holdout_count:] = "holdout"
    return pd.DataFrame({"sample_id": sample_ids, "split": split})


def _build_feature_catalog(
    *,
    spec: SyntheticDGPSpec,
    true_interactions: frozenset[tuple[str, str]],
) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for input_idx in range(spec.n_inputs):
        records.append(
            {
                "feature_name": f"x{input_idx}",
                "feature_type": "first_order",
                "origin": spec.dgp_family,
            }
        )

    pair_cap = min(spec.n_inputs, 50)
    interaction_pairs = {
        _ordered_feature_pair(left_idx, right_idx)
        for left_idx, right_idx in combinations(range(pair_cap), 2)
    }
    interaction_pairs.update(true_interactions)
    for left_name, right_name in sorted(interaction_pairs, key=_pair_sort_key):
        records.append(
            {
                "feature_name": f"{left_name}:{right_name}",
                "feature_type": "interaction",
                "origin": spec.dgp_family,
            }
        )

    for input_idx in range(spec.n_inputs):
        for transform_name in _CATALOG_NONLINEAR_TRANSFORMS:
            records.append(
                {
                    "feature_name": f"{transform_name}_x{input_idx}",
                    "feature_type": "transformation",
                    "origin": spec.dgp_family,
                }
            )

    return pd.DataFrame.from_records(
        records,
        columns=["feature_name", "feature_type", "origin"],
    )


def _ar1_correlation_matrix(n_inputs: int, rho: float) -> np.ndarray:
    offsets = np.abs(np.subtract.outer(np.arange(n_inputs), np.arange(n_inputs)))
    return np.power(rho, offsets, dtype=np.float64)


def _ordered_feature_pair(left_idx: int, right_idx: int) -> tuple[str, str]:
    left_name = f"x{left_idx}"
    right_name = f"x{right_idx}"
    return (left_name, right_name) if left_idx < right_idx else (right_name, left_name)


def _pair_sort_key(pair: tuple[str, str]) -> tuple[int, int]:
    return (_feature_index(pair[0]), _feature_index(pair[1]))


def _feature_name_sort_key(name: str) -> tuple[int, str]:
    return (_feature_index(name.split("_x", 1)[1]), name)


def _feature_index(feature_name: str) -> int:
    return int(feature_name.removeprefix("x"))


def _evaluate_transform(transform_name: str, values: np.ndarray) -> np.ndarray:
    transform_map: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "sin": lambda arr: np.sin(2.0 * pi * arr),
        "sq": lambda arr: np.square(arr) - (1.0 / 3.0),
        "exp": lambda arr: np.exp(-3.0 * arr) - ((1.0 - exp(-3.0)) / 3.0),
        "log1p": lambda arr: np.log1p(arr) - ((2.0 * np.log(2.0)) - 1.0),
    }
    return transform_map[transform_name](values)


def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer; got {value!r}")


def _validate_range(value: float, field_name: str, lower: float, upper: float) -> None:
    if not lower <= value <= upper:
        raise ValueError(f"{field_name} must lie in [{lower}, {upper}]; got {value!r}")


__all__ = [
    "DGPTrueSupport",
    "SyntheticDataset",
    "SyntheticDGPSpec",
    "generate_calibrated_structure_synthetic",
    "generate_pure_synthetic",
]
